"""
banc_v10.py — outils du banc pour le firmware v10 (firmware/mcu6/simpleserial-aes).

Contenu :
  * chemins du projet et du disque de traces (vérification que le disque est bien monté) ;
  * connexion, programmation de la carte, lecture de la version du firmware (commande 'E') ;
  * réglage des délais à moyenne flottante (commande 'J') ;
  * captures : campagne à clé fixe (attaque) et campagne à clés aléatoires (apprentissage d'un CNN) ;
  * calibration d'une intensité de délai pour un surcoût visé ;
  * vérifications rapides : fuite de l'AddRoundKey(0) à instant fixe, distribution des durées ;
  * évaluation par l'attaquant du mémoire (verify_n, fenêtre (300, 24000)), parallélisée par octet.

Les modules d'analyse (sca_metrics, verify_n, parego_v2, ...) sont des copies de projets2 : le dossier projets2 et
ses campagnes v9 restent intacts (lecture seule, état « avant correction » du mémoire).
"""
import json
import multiprocessing as mp
import os
import re
import time

import numpy as np

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
PROJET = "/home/khadim/chipwhisperer/projet"
ANCIEN = "/home/khadim/chipwhisperer/projets2"                   # campagnes et historique v9 (lecture seule)
MONITOR = "/home/khadim/chipwhisperer/monitor"                    # monitor_collect.py (liaison série)
FW_DIR = "/home/khadim/chipwhisperer/firmware/mcu6/simpleserial-aes"
HEX = os.path.join(FW_DIR, "simpleserial-aes-CWLITEARM.hex")
DISQUE = "/media/khadim/2A53DBCE3700DFDA"                         # point de montage du disque externe
TRACE_ROOT = os.path.join(DISQUE, "trace", "v10")                 # toutes les nouvelles traces

KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
WIN = (300, 24000)                                                # fenêtre d'évaluation (t < 300 : chargement du texte clair)
MAX_SHIFT = 3000
FM_DEFAUT = 2                                                     # b = a >> 2 (Coron & Kizhvatov 2010)
FW_VERSION_ATTENDUE = 13                                          # v13 (escalier calibré ; v12 : arbre unique, compteurs de faute persistants)


def verifier_disque(min_go=10.0):
    """Refuse d'écrire si le disque externe n'est pas monté : sinon les traces iraient sur le disque système
    (dans le dossier vide du point de montage)."""
    if not os.path.ismount(DISQUE):
        raise RuntimeError("Le disque %s n'est pas monté : branchez-le avant de capturer." % DISQUE)
    os.makedirs(TRACE_ROOT, exist_ok=True)
    st = os.statvfs(TRACE_ROOT)
    libre = st.f_bavail * st.f_frsize / 1e9
    if libre < min_go:
        raise RuntimeError("Espace libre insuffisant sur %s : %.1f Go (minimum %.0f Go)." % (DISQUE, libre, min_go))
    return libre


def dossier(*parts):
    p = os.path.join(TRACE_ROOT, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def empreinte_firmware(hex_path=HEX):
    """Empreinte (12 caractères du SHA-256) du fichier .hex programmé : identifie le build du firmware."""
    import hashlib
    with open(hex_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _marquer_build(path, build):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "build.txt"), "w") as f:
        f.write(build)


def archiver_si_autre_build(path, build):
    """Si une campagne existe déjà sous ce nom mais vient d'un AUTRE build (ou d'un build inconnu), elle est
    renommée <nom>_build_<ancien> au lieu d'être réutilisée ou effacée : rien n'est perdu, et la nouvelle
    campagne est capturée avec le firmware actuel."""
    if not os.path.isdir(path):
        return None
    mp_, bp_ = os.path.join(path, "meta.json"), os.path.join(path, "build.txt")
    if os.path.exists(mp_):
        ancien = json.load(open(mp_)).get("build")
    elif os.path.exists(bp_):
        ancien = open(bp_).read().strip()            # capture interrompue : reprise si même build
    else:
        ancien = None
    if ancien == build:
        return None
    dst = "%s_build_%s" % (path.rstrip("/"), ancien or "inconnu")
    k = 1
    while os.path.exists(dst):
        dst = "%s_build_%s_%d" % (path.rstrip("/"), ancien or "inconnu", k); k += 1
    os.rename(path, dst)
    print("campagne d'un autre build archivée : %s -> %s" % (os.path.basename(path), os.path.basename(dst)))
    return dst


# --------------------------------------------------------------------------
# Connexion et firmware
# --------------------------------------------------------------------------
class NoiseTarget:
    """Mêmes réglages d'oscilloscope que toutes les campagnes v9 (comparabilité des traces)."""
    def __init__(self, scope, target):
        self.scope, self.target = scope, target
        scope.clock.adc_src = "clkgen_x1"
        scope.adc.samples = 24400
        scope.gain.gain = 25
        scope.adc.basic_mode = "rising_edge"
        scope.trigger.triggers = "tio4"
        self.reset()

    def reset(self):
        self.scope.io.nrst = "low"; time.sleep(0.05)
        self.scope.io.nrst = "high_z"; time.sleep(0.2)
        try:
            self.target.read(timeout=100)
        except Exception:
            pass


def connecter(programmer=True, hex_path=HEX):
    import chipwhisperer as cw
    scope = cw.scope()
    target = cw.target(scope, cw.targets.SimpleSerial)
    scope.default_setup()
    if programmer:
        if not os.path.exists(hex_path):
            raise FileNotFoundError("%s absent : compilez le firmware v10 d'abord." % hex_path)
        cw.program_target(scope, cw.programmers.STM32FProgrammer, hex_path)
    return NoiseTarget(scope, target)


def _lire_ligne(nt, cmd, data, lettre, n_octets):
    import monitor_collect as mc
    import sca_capture as sc
    mc._flush(nt.target, 50)
    nt.target.simpleserial_write(cmd, bytearray(data))
    buf = sc._read_until_z(nt.target)
    m = re.search(r"%s([0-9A-Fa-f]{%d})" % (lettre, 2 * n_octets), buf)
    if not m:
        raise RuntimeError("pas de réponse '%s' à la commande '%s' (reçu : %r)" % (lettre, cmd, buf[:80]))
    return bytes.fromhex(m.group(1))


def info_firmware(nt, verbose=True):
    """Commande 'E' : version, graine ADC, changements du bit de poids faible de l'ADC, fm_shift."""
    b = _lire_ligne(nt, "E", [0], "e", 4)
    info = dict(version=b[0], graine_adc=bool(b[1]), changements_lsb=b[2], fm_shift=b[3])
    if verbose:
        print("firmware v%d | graine ADC : %s (%d changements du bit de poids faible sur 510) | moyenne flottante : b = a >> %d"
              % (info["version"], "oui" if info["graine_adc"] else "NON (graine de repli)", info["changements_lsb"],
                 info["fm_shift"]))
        if info["version"] != FW_VERSION_ATTENDUE:
            print("ATTENTION : la carte n'exécute pas le firmware v%d." % FW_VERSION_ATTENDUE)
        if info["graine_adc"] and info["changements_lsb"] < 50:
            print("ATTENTION : bit de poids faible de l'ADC peu variable : graine de faible entropie.")
    return info


def regler_fm(nt, shift=FM_DEFAUT):
    import monitor_collect as mc
    mc._cmd(nt.target, "J", bytearray([int(shift)]))


def setup_v10(cfg=None, fm_shift=FM_DEFAUT):
    """Fonction de réglage pour sca_capture.capture_campaign : config de bruit (ou aucune) + moyenne flottante.
    make_setup remet d'abord la cible à zéro (aucun forçage, aucune intensité brute, toutes rondes)."""
    import sca_capture as sc
    base = cfg.setup() if cfg is not None else sc.make_setup()

    def _s(nt):
        base(nt)
        regler_fm(nt, fm_shift)
    return _s


def verifier_scope(nt, ref_meta):
    """Compare le réglage de l'oscilloscope à celui d'une campagne de référence (meta.json)."""
    import sca_capture as sc
    ref = json.load(open(ref_meta)).get("scope") or {}
    cur = sc.scope_info(nt)

    def egal(a, b, tol=1e-4):
        try:
            return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            return str(a) == str(b)
    diff = {k: (cur.get(k), ref[k]) for k in ref if k != "trigger" and not egal(cur.get(k), ref[k])}
    if diff:
        print("Réglage de l'oscilloscope différent de la référence (actuel, référence) :", diff)
    else:
        print("Réglage de l'oscilloscope identique à la référence.")
    return diff


# --------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------
def capturer(nt, nom, n_traces, cfg=None, fm_shift=FM_DEFAUT, sous_dossier="campagnes", seed=0, key=KEY,
             meta=None, chunk=1000):
    """Campagne à clé FIXE (jeu d'attaque). Graine 0 = mêmes textes clairs que traces/comp20k (v9)."""
    import sca_capture as sc
    verifier_disque()
    build = empreinte_firmware()
    archiver_si_autre_build(os.path.join(dossier(sous_dossier), nom), build)
    _marquer_build(os.path.join(dossier(sous_dossier), nom), build)
    m = dict(firmware=FW_VERSION_ATTENDUE, build=build, fm_shift=int(fm_shift),
             config=(cfg.to_dict() if cfg is not None else dict(comps=[], raw=[], mask=0, name="|m000")))
    m.update(meta or {})
    return sc.capture_campaign(nt, nom, n_traces, key, seed=seed, outdir=dossier(sous_dossier), chunk=chunk,
                               setup=setup_v10(cfg, fm_shift), meta=m)


def capturer_cles_aleatoires(nt, nom, n_traces, cfg=None, fm_shift=FM_DEFAUT, sous_dossier="apprentissage",
                             seed=1, chunk=1000, fenetre=None):
    """Campagne à clés ALÉATOIRES (une clé par trace), pour l'apprentissage d'un attaquant profilé (CNN).
    Les campagnes à clé fixe servent ensuite de jeu d'attaque : on n'apprend jamais sur la clé attaquée.
    fenetre=(a, b) : n'enregistre que les échantillons a..b-1 (réduit la taille sur disque).
    Reprise automatique par blocs."""
    import monitor_collect as mc
    import sca_capture as sc
    from Crypto.Cipher import AES
    verifier_disque()
    path = os.path.join(dossier(sous_dossier), nom)
    build = empreinte_firmware()
    archiver_si_autre_build(path, build)
    _marquer_build(path, build)
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 256, (n_traces, 16), dtype=np.uint8)
    pts = rng.integers(0, 256, (n_traces, 16), dtype=np.uint8)
    n_chunks = (n_traces + chunk - 1) // chunk
    cpath = lambda i: os.path.join(path, "chunk_%05d.npz" % i)
    todo = [i for i in range(n_chunks) if not os.path.exists(cpath(i))]
    if os.path.exists(os.path.join(path, "meta.json")):     # capture terminée (ses blocs sont effacés à la fin)
        print("[%s] déjà complète" % nom)
        return path
    mc.prepare_target(nt.target)
    setup_v10(cfg, fm_shift)(nt)
    t0, done = time.time(), 0
    for i in todo:
        a, b = i * chunk, min(n_traces, (i + 1) * chunk)
        trs, cts, trg = [], [], []
        bad = 0
        for j in range(a, b):
            mc._cmd(nt.target, "k", bytearray(keys[j]))
            tr, ct, tg = sc.capture_one(nt, pts[j], with_trig=True)
            if fenetre is not None:
                tr = tr[fenetre[0]:fenetre[1]]
            exp = np.frombuffer(AES.new(bytes(keys[j]), AES.MODE_ECB).encrypt(bytes(pts[j])), np.uint8)
            bad += int((ct != exp).any())
            trs.append(tr); cts.append(ct); trg.append(tg)
        np.savez(cpath(i), traces=np.stack(trs), cts=np.stack(cts), trig=np.array(trg, np.uint32), ct_bad=bad)
        done += b - a
        el = time.time() - t0
        print("[%s] bloc %d/%d | chiffrés faux : %d | %.0f ms/trace | reste ~%.0f min"
              % (nom, i + 1, n_chunks, bad, el / done * 1000, el / done * (len(todo) * chunk - done) / 60))
    first = np.load(cpath(0))
    T = first["traces"].shape[1]
    mm = np.lib.format.open_memmap(os.path.join(path, "traces.npy"), mode="w+", dtype=np.int16, shape=(n_traces, T))
    cts_all = np.empty((n_traces, 16), np.uint8); trig_all = np.zeros(n_traces, np.uint32); bad_total = 0
    for i in range(n_chunks):
        z = np.load(cpath(i)); a = i * chunk
        mm[a:a + len(z["traces"])] = z["traces"]; cts_all[a:a + len(z["cts"])] = z["cts"]
        trig_all[a:a + len(z["trig"])] = z["trig"]; bad_total += int(z["ct_bad"])
    mm.flush(); del mm
    np.save(os.path.join(path, "pts.npy"), pts); np.save(os.path.join(path, "keys.npy"), keys)
    np.save(os.path.join(path, "cts.npy"), cts_all); np.save(os.path.join(path, "trig.npy"), trig_all)
    info = dict(name=nom, n_traces=int(n_traces), n_samples=int(T), key="aleatoire", seed=int(seed),
                fenetre=list(fenetre) if fenetre else None, ct_errors=int(bad_total),
                trig_mean=float(trig_all.mean()), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                raw_to_float="trace = raw/1024 - 0.5", scope=sc.scope_info(nt), firmware=FW_VERSION_ATTENDUE,
                build=build, fm_shift=int(fm_shift),
                config=(cfg.to_dict() if cfg is not None else dict(comps=[], raw=[], mask=0, name="|m000")))
    json.dump(info, open(os.path.join(path, "meta.json"), "w"), indent=2)
    for i in range(n_chunks):
        os.remove(cpath(i))
    print("[%s] OK : %d traces x %d échantillons | chiffrés faux : %d | %s" % (nom, n_traces, T, bad_total, path))
    return path


def mesurer_durees(nt, cfg=None, n=200, fm_shift=FM_DEFAUT, pt=None):
    """Durées (cycles, trig_count) de n chiffrements, sans enregistrer de trace. pt fixe : seule la variabilité
    due au bruit apparaît."""
    import monitor_collect as mc
    import sca_capture as sc
    mc.prepare_target(nt.target)
    mc._cmd(nt.target, "k", bytearray(KEY))
    setup_v10(cfg, fm_shift)(nt)
    rng = np.random.default_rng(123)
    out = []
    for _ in range(n):
        p = pt if pt is not None else rng.integers(0, 256, 16, dtype=np.uint8)
        out.append(sc.capture_one(nt, p, with_trig=True)[2])
    return np.array(out, dtype=float)


def calibrer(nt, fabrique, cible_pct, base_cyc, bornes=(1, 2000), n=150, fm_shift=FM_DEFAUT, tol_pct=1.0):
    """Recherche par dichotomie de l'intensité brute r telle que le surcoût de fabrique(r) vaille cible_pct.
    fabrique : r -> noise_opt.Config. Retourne (r, surcoût mesuré)."""
    lo, hi = bornes
    best = None
    for _ in range(12):
        r = (lo + hi) // 2
        pct = 100 * (mesurer_durees(nt, fabrique(r), n, fm_shift).mean() / base_cyc - 1)
        print("   intensité %4d -> surcoût %+.1f %%" % (r, pct))
        if best is None or abs(pct - cible_pct) < abs(best[1] - cible_pct):
            best = (r, pct)
        if abs(pct - cible_pct) <= tol_pct or hi - lo <= 1:
            break
        if pct < cible_pct:
            lo = r
        else:
            hi = r
    print("intensité retenue : %d (surcoût %+.1f %%, cible %+.1f %%)" % (best[0], best[1], cible_pct))
    return best


# --------------------------------------------------------------------------
# Vérifications rapides
# --------------------------------------------------------------------------
def rapport_durees(path, base_cyc):
    t = np.load(os.path.join(path, "trig.npy")).astype(float)
    r = dict(moyenne=float(t.mean()), ecart_type=float(t.std()), p1=float(np.percentile(t, 1)),
             p99=float(np.percentile(t, 99)), surcout_pct=float(100 * (t.mean() / base_cyc - 1)))
    print("%s : durée %.0f ± %.0f cycles (1 %%–99 %% : %.0f–%.0f) | surcoût %+.1f %%"
          % (os.path.basename(path), r["moyenne"], r["ecart_type"], r["p1"], r["p99"], r["surcout_pct"]))
    return r


def verif_ark0(path, ref_path, n=2000, fenetre=(300, 3000), verbose=True):
    """Fuite de l'AddRoundKey(0) : pour chaque octet, instant de fuite maximale (modèle HW(p xor k)) dans la
    référence sans bruit, puis CPA sur 256 hypothèses À CET INSTANT FIXE dans la campagne testée (n traces).
    Une protection correcte donne 0 octet trouvé. Donne aussi le |rho| maximal de la bonne clé dans la fenêtre,
    à comparer au seuil de significativité 4/sqrt(n)."""
    import sca_metrics as sm
    import verify_n as vn
    ref = sm._load(ref_path)
    d = sm._load(path)
    nr = min(2000, len(ref["traces"]))
    rho_ref = sm.corr_profile(ref["traces"], ref["pts"][:], ref["key"], tuple(range(16)), np.arange(nr), fenetre,
                              1, model="xor")
    t_fix = fenetre[0] + np.argmax(np.abs(rho_ref), axis=1)
    n = min(n, len(d["traces"]))
    idx = np.arange(n)
    rho_cur = sm.corr_profile(d["traces"], d["pts"], d["key"], tuple(range(16)), idx, fenetre, 1, model="xor")
    key = np.asarray(d["key"], np.uint8)
    rows = []
    for b in range(16):
        H = vn._hyp_all(d["pts"][:n, b], "xor")
        f = np.asarray(d["traces"][:n, int(t_fix[b])], dtype=np.float64)
        s = np.abs(vn._scores(H, f, idx))
        rows.append(dict(octet=b, t=int(t_fix[b]), rang=int((s > s[key[b]]).sum()),
                         rho_fixe=float(s[key[b]]), rho_max_fenetre=float(np.abs(rho_cur[b]).max())))
    thr = 4 / np.sqrt(n)
    # maximum attendu du bruit seul sur 16 octets x W instants (tests multiples) : sqrt(2 ln(16 W)) / sqrt(n)
    thr_max = np.sqrt(2 * np.log(16 * (fenetre[1] - fenetre[0]))) / np.sqrt(n)
    trouves = sum(r["rang"] == 0 for r in rows)
    if verbose:
        print("%s, %d traces : %d octet(s) sur 16 trouvés à instant fixe | |rho| max de la bonne clé : %.3f "
              "(seuil 4/sqrt(n) = %.3f ; maximum attendu du bruit seul sur la fenêtre ≈ %.3f)"
              % (os.path.basename(path), n, trouves, max(r["rho_max_fenetre"] for r in rows), thr, thr_max))
    return dict(octets_trouves=int(trouves), seuil=float(thr), seuil_max_bruit=float(thr_max), detail=rows)


# --------------------------------------------------------------------------
# Évaluation par l'attaquant du mémoire (verify_n), parallélisée par octet
# --------------------------------------------------------------------------
def nproc_sur(go_par_processus=3.0):
    """Nombre de processus de calcul compatible avec la mémoire disponible (environ 3 Go par processus pour une
    campagne de 20 000 traces) et le nombre de cœurs."""
    dispo = None
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                dispo = int(l.split()[1]) / 1e6
    except OSError:
        pass
    n_mem = int(dispo // go_par_processus) if dispo else 4
    return max(1, min((os.cpu_count() or 2) - 1, n_mem))


_G = {}


def _best_octet(b):
    import sca_metrics as sm
    import verify_n as vn
    return b, vn.best_attackers(_G["d"], _G["clean"], (b,), MAX_SHIFT, sm.MODELS, 0, _G["win"]).get(b)


def _resume(out, trig_mean):
    ns, bsr, bits = out["ns"], out["byte_sr"], out["effort_bits"]
    premier = lambda pred: next((int(n) for n in ns if pred(n)), None)
    r = dict(ns=[int(n) for n in ns], trials=int(out["trials"]),
             octets=[float(out["mean_bytes"][n]) for n in ns],
             octet_facile_sr=[float(bsr[n].max()) for n in ns], octet_dur_sr=[float(bsr[n].min()) for n in ns],
             cle_sr=[float(out["full_sr"][n]) for n in ns],
             effort_med=[float(np.median(bits[n])) for n in ns],
             effort_p10=[float(np.percentile(bits[n], 10)) for n in ns],
             effort_p90=[float(np.percentile(bits[n], 90)) for n in ns],
             n_effort32=premier(lambda n: np.mean(bits[n] <= 32) >= 0.9), trig_mean=float(trig_mean))
    r["X_median"] = next((n for n, e in zip(r["ns"], r["effort_med"]) if e <= 32), None)
    return r


def evaluer(path, clean, ns=(20, 50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 4000, 6000, 8000), trials=100,
            nproc=None, win=WIN, refaire=False):
    """Attaque directe (même attaquant et même protocole que la cellule 116 du notebook v8). Résultat écrit dans
    <campagne>_resultat_w300.json, à côté de la campagne ; reprise si le fichier existe déjà."""
    import sca_metrics as sm
    import verify_n as vn
    jp = path.rstrip("/") + "_resultat_w%d.json" % win[0]
    if os.path.exists(jp) and not refaire:
        return json.load(open(jp))
    d = sm._load(path)
    _G.update(d=d, clean=clean, win=tuple(win))
    nproc = int(nproc or nproc_sur())
    t0 = time.time()
    # ProcessPoolExecutor : si un processus est tué (manque de mémoire), une erreur est levée au lieu d'un blocage
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(nproc, 16), mp_context=mp.get_context("fork")) as ex:
        best = {b: v for b, v in ex.map(_best_octet, range(16)) if v is not None}
    ns = [n for n in ns if n <= len(d["traces"]) // 2]
    out = vn.success_curve(path, clean, ns=ns, trials=trials, max_shift=MAX_SHIFT, window=win, best=best, verbose=False)
    t = np.load(os.path.join(path, "trig.npy")).astype(float).mean()
    r = _resume(out, t)
    r.update(chemin=path, win=list(win), duree_calcul_s=round(time.time() - t0))
    json.dump(r, open(jp, "w"), indent=1)
    print("%s : X médian = %s | effort médian à n max = %.1f bits | %.0f min de calcul"
          % (os.path.basename(path), r["X_median"] or "> %d" % ns[-1], r["effort_med"][-1], r["duree_calcul_s"] / 60))
    return r


def tableau(resultats, base_cyc):
    """resultats : {nom: dict de evaluer()} -> lignes imprimées et liste de dicts."""
    rows = []
    print("%-30s %9s %9s %9s %12s" % ("condition", "surcoût", "X (méd.)", "X (90 %)", "bits à n max"))
    for nom, r in resultats.items():
        sur = 100 * (r["trig_mean"] / base_cyc - 1)
        xm = r.get("X_median") or "> %d" % r["ns"][-1]
        x9 = r.get("n_effort32") or "> %d" % r["ns"][-1]
        rows.append(dict(condition=nom, surcout_pct=round(sur, 1), X_median=xm, X_p90=x9,
                         effort_bits_nmax=round(r["effort_med"][-1], 1), n_max=r["ns"][-1]))
        print("%-30s %+8.1f%% %9s %9s %12.1f" % (nom, sur, xm, x9, r["effort_med"][-1]))
    return rows
