"""
glitch_v10.py — étape 3.3 : balayages de glitchs d'horloge sur l'AES du firmware v10.1, avec enregistrement des
signaux de faute (lignes 'G', 'H', 'P') pour le classifieur de l'étape 4. Aucune récupération de clé ici : on
mesure seulement les effets des glitchs et ce que la carte en détecte.

Une campagne = alternance de blocs « normaux » (horloge propre, pas de glitch) et de blocs « glitch » (réglages
tirés au hasard dans les plages données). Chaque essai est enregistré (une ligne JSON) : reprise automatique.

Classe d'un essai :
  normal    chiffré correct
  detecte   la carte a vu la faute (compteur de rondes ou désaccord de redondance) : sortie non livrée
  exploit   chiffré faux NON détecté, motif de 4 octets en diagonale (faute DFA de Piret-Quisquater)
  faute     chiffré faux NON détecté, autre motif
  reset     pas de réponse (plantage) : la cible est redémarrée
Le signal « durée nette anormale » n'agit pas sur la sortie : il est seulement enregistré, pour mesurer
après coup ce qu'il aurait détecté (seuil choisi sur les blocs normaux).
"""
import json
import logging
import os
import time

import numpy as np

import fautes_v10 as fv

DFA_DIAGONALS = (frozenset({0, 7, 10, 13}), frozenset({1, 4, 11, 14}),
                 frozenset({2, 5, 8, 15}), frozenset({3, 6, 9, 12}))
CLASSES = ("normal", "detecte", "exploit", "faute", "reset")
MODES = {"sans_redondance": 0, "redondance_toujours": 1, "redondance_auto": 2}


def classer(ct, ref, g):
    if ct is None or g is None:
        return "reset"
    if g["round_err_op"] or g["mismatch"]:
        return "detecte"
    diff = frozenset(i for i in range(16) if ct[i] != ref[i])
    if not diff:
        return "normal"
    return "exploit" if (len(diff) == 4 and diff in DFA_DIAGONALS) else "faute"


def setup_clock_glitch(scope, repeat=1):
    """Glitch d'horloge (réglages du tutoriel ChipWhisperer Fault 1.1, CW-Lite)."""
    scope.glitch.clk_src = "clkgen"
    scope.glitch.output = "clock_xor"
    scope.glitch.trigger_src = "ext_single"
    scope.glitch.repeat = int(repeat)
    scope.io.hs2 = "glitch"


def horloge_propre(scope):
    scope.io.hs2 = "clkgen"
    try:
        scope.glitch.trigger_src = "manual"
    except Exception:
        pass


def _preparer(nt, key, mode, adaptatif, effacer=False):
    import banc_v10 as bv
    import monitor_collect as mc
    mc.prepare_target(nt.target)
    mc._cmd(nt.target, "k", bytearray(key))
    if effacer:
        bv._lire_ligne(nt, "H", [1], "H", 8)
    fv.rapport_fautes(nt, mode=mode)
    if adaptatif:
        mc._cmd(nt.target, "a", bytearray([1]))       # politique embarquée (plancher + règle x6)


def _reboot(nt):
    s = nt.scope
    s.io.nrst = "low"; time.sleep(0.05)
    s.io.nrst = "high_z"; time.sleep(0.10)
    try:
        nt.target.flush()
    except Exception:
        pass


def _essai(nt, pt):
    import sca_capture as sc
    nt.scope.arm()
    nt.target.simpleserial_write("p", bytearray(pt))
    if nt.scope.capture():
        return None
    try:
        buf = sc._read_until_z(nt.target, timeout_s=0.5)
    except Exception:
        return None
    m = fv.RE_R.search(buf)
    p = fv.RE_P.search(buf)
    return dict(ct=bytes.fromhex(m.group(1)) if m else None, g=fv.parse_g(buf), h=fv.parse_h(buf),
                plog=p.group(1) if p else None)


def campagne(nt, key, nom, outdir, mode="sans_redondance", n_glitch=3000, bloc_glitch=200, bloc_normal=50,
             width=(-20, 20), offset=(-48, 48), ext=(0, 50000), repeat=1, adaptatif=True, seed=0, zones=None, repeats=None):
    """Reprenable : les essais déjà présents dans <outdir>/<nom>.jsonl ne sont pas refaits.
    zones : liste de (largeur_min, largeur_max, décalage_min, décalage_max) ; chaque essai tire une zone au hasard
    puis un réglage uniforme dans cette zone (campagne ciblée sur les régions productives de l'exploration)."""
    import sca_capture as sc
    for lg in ("ChipWhisperer Target", "ChipWhisperer Glitch", "ChipWhisperer Scope"):
        logging.getLogger(lg).setLevel(logging.CRITICAL)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nom + ".jsonl")
    deja = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    rng = np.random.default_rng(seed)
    md = MODES[mode]
    # plan complet (déterministe par la graine) : blocs normal / glitch alternés
    plan, ng = [], 0
    while ng < n_glitch:
        plan += [False] * bloc_normal
        k = min(bloc_glitch, n_glitch - ng); plan += [True] * k; ng += k
    plan += [False] * bloc_normal
    if zones:
        zs = [zones[int(rng.integers(len(zones)))] for _ in plan]
        reglages = [(float(rng.uniform(z[0], z[1])), float(rng.uniform(z[2], z[3])), int(rng.integers(*ext))) for z in zs]
    else:
        reglages = [(float(rng.uniform(*width)), float(rng.uniform(*offset)), int(rng.integers(*ext))) for _ in plan]
    reps = [int(repeats[int(rng.integers(len(repeats)))]) if repeats else int(repeat) for _ in plan]
    pts = rng.integers(0, 256, (len(plan), 16), dtype=np.uint8)
    refs = sc.expected_ct(bytes(key), pts)                     # chiffrés attendus (même AES que le banc)
    nt.scope.adc.timeout = 0.2
    setup_clock_glitch(nt.scope, repeat)
    _reboot(nt); _preparer(nt, key, md, adaptatif, effacer=(deja == 0))
    t0, cnt, etat = time.time(), {c: 0 for c in CLASSES}, None
    with open(path, "a") as f:
        for i in range(deja, len(plan)):
            g_on = plan[i]
            if g_on != etat:                                   # changement de bloc
                if g_on:
                    setup_clock_glitch(nt.scope, repeat)
                else:
                    horloge_propre(nt.scope)
                etat = g_on
            w, o, e = reglages[i]
            if g_on:
                nt.scope.glitch.repeat = reps[i]
                nt.scope.glitch.width, nt.scope.glitch.offset, nt.scope.glitch.ext_offset = w, o, e
            r = _essai(nt, pts[i])
            ref = bytes(refs[i])
            cl = classer(r["ct"] if r else None, ref, r["g"] if r else None)
            cnt[cl] += 1
            rec = dict(i=i, glitch=bool(g_on), width=w if g_on else None, offset=o if g_on else None,
                       repeat=reps[i] if g_on else None,
                       ext=e if g_on else None, classe=cl, t=round(time.time() - t0, 3),
                       ndiff=(sum(a != b for a, b in zip(r["ct"], ref)) if (r and r["ct"]) else None),
                       g=r["g"] if r else None, h=r["h"] if r else None, plog=r["plog"] if r else None)
            f.write(json.dumps(rec) + "\n")
            if cl == "reset":
                _reboot(nt); _preparer(nt, key, md, adaptatif)
                if g_on:
                    setup_clock_glitch(nt.scope, repeat)
            if (i + 1) % 500 == 0:
                f.flush()
                print("[%s] %d/%d essais | %s | %.1f essais/s" % (nom, i + 1, len(plan),
                      " ".join("%s %d" % (c, cnt[c]) for c in CLASSES), (i + 1 - deja) / (time.time() - t0)))
    horloge_propre(nt.scope)
    return path


def charger(path):
    return [json.loads(l) for l in open(path)]


def analyse(recs, net_ref=None, seuil=None):
    """Taux par classe (essais avec / sans glitch) et pouvoir de détection de chaque signal.
    net_ref, seuil : durée nette de référence et écart toléré (choisis sur les essais normaux sans glitch)."""
    norm = [r for r in recs if not r["glitch"] and r["g"]]
    nets = np.array([r["g"]["net_cycles"] for r in norm], float)
    if net_ref is None:
        net_ref = float(np.median(nets))
    if seuil is None:
        seuil = float(np.max(np.abs(nets - net_ref))) + 5.0      # aucune fausse alerte sur les blocs normaux
    anormal = lambda r: bool(r["g"]) and abs(r["g"]["net_cycles"] - net_ref) > seuil
    out = dict(net_ref=net_ref, seuil=seuil, n_normaux=len(norm),
               fausses_alertes_duree=int(sum(anormal(r) for r in norm)),
               classes_sans_glitch={c: sum(r["classe"] == c for r in recs if not r["glitch"]) for c in CLASSES},
               classes_glitch={c: sum(r["classe"] == c for r in recs if r["glitch"]) for c in CLASSES})
    eff = [r for r in recs if r["glitch"] and r["classe"] in ("detecte", "exploit", "faute")]
    out["fautes_effectives"] = len(eff)
    if eff:
        out["vues_par_ronde"] = int(sum(bool(r["g"]["round_err_op"]) for r in eff))
        out["vues_par_redondance"] = int(sum(bool(r["g"]["mismatch"]) for r in eff))
        out["vues_par_duree"] = int(sum(anormal(r) for r in eff))
        out["non_vues"] = int(sum(not (r["g"]["round_err_op"] or r["g"]["mismatch"] or anormal(r)) for r in eff))
    return out


def exploration(nt, key, outdir, nom="exploration", widths=range(-40, 41, 8), offsets=range(-45, 46, 9),
                repeats=(1, 5, 10), n_point=8, ext=(0, 50000), mode="sans_redondance", seed=7):
    """Recherche des réglages efficaces (étape standard du tutoriel ChipWhisperer) : grille largeur x décalage x
    répétition, n_point essais par point à des instants (ext_offset) aléatoires. Reprenable (JSONL)."""
    import sca_capture as sc
    for lg in ("ChipWhisperer Target", "ChipWhisperer Glitch", "ChipWhisperer Scope"):
        logging.getLogger(lg).setLevel(logging.CRITICAL)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, nom + ".jsonl")
    deja = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    rng = np.random.default_rng(seed)
    grille = [(float(w), float(o), int(r)) for r in repeats for w in widths for o in offsets if w != 0]
    plan = [(w, o, r, int(rng.integers(*ext))) for (w, o, r) in grille for _ in range(n_point)]
    pts = rng.integers(0, 256, (len(plan), 16), dtype=np.uint8)
    refs = sc.expected_ct(bytes(key), pts)
    md = MODES[mode]
    nt.scope.adc.timeout = 0.2
    _reboot(nt); _preparer(nt, key, md, adaptatif=False, effacer=True)
    setup_clock_glitch(nt.scope, 1)
    t0, cnt = time.time(), {c: 0 for c in CLASSES}
    with open(path, "a") as f:
        for i in range(deja, len(plan)):
            w, o, rp, e = plan[i]
            nt.scope.glitch.repeat = rp
            nt.scope.glitch.width, nt.scope.glitch.offset, nt.scope.glitch.ext_offset = w, o, e
            r = _essai(nt, pts[i])
            cl = classer(r["ct"] if r else None, bytes(refs[i]), r["g"] if r else None)
            cnt[cl] += 1
            dev = (r["g"]["net_cycles"] if (r and r["g"]) else None)
            f.write(json.dumps(dict(i=i, width=w, offset=o, repeat=rp, ext=e, classe=cl, net=dev,
                                    ndiff=(sum(a != b for a, b in zip(r["ct"], bytes(refs[i]))) if (r and r["ct"]) else None))) + "\n")
            if cl == "reset":
                _reboot(nt); _preparer(nt, key, md, adaptatif=False, effacer=True)
                setup_clock_glitch(nt.scope, rp)
            if (i + 1) % 500 == 0:
                f.flush()
                print("[%s] %d/%d | %s | %.1f essais/s" % (nom, i + 1, len(plan),
                      " ".join("%s %d" % (c, cnt[c]) for c in CLASSES), (i + 1 - deja) / (time.time() - t0)))
    horloge_propre(nt.scope)
    return path


def resume_exploration(path, net_ref=None, seuil=25, top=12):
    """Points de la grille classés par taux de fautes NON reset (faute + exploit + détecté + durée anormale)."""
    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, reset=0, faute=0, exploit=0, detecte=0, duree=0))
    if net_ref is None:                                  # référence = médiane des essais sans effet (mêmes réglages de banc)
        net_ref = float(np.median([json.loads(l)["net"] for l in open(path)
                                   if json.loads(l)["classe"] == "normal" and json.loads(l)["net"] is not None]))
    for l in open(path):
        r = json.loads(l)
        a = agg[(r["width"], r["offset"], r["repeat"])]
        a["n"] += 1
        a[r["classe"]] = a.get(r["classe"], 0) + 1
        if r["classe"] == "normal" and r["net"] is not None and abs(r["net"] - net_ref) > seuil:
            a["duree"] += 1
    rows = sorted(agg.items(), key=lambda kv: -(kv[1]["faute"] + kv[1]["exploit"] + kv[1]["detecte"] + kv[1]["duree"]
                                                 + kv[1]["reset"]))
    tot = {k: sum(v[k] for v in agg.values()) for k in ("n", "reset", "faute", "exploit", "detecte", "duree")}
    return rows[:top], tot
