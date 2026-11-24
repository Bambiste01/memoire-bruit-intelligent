"""
parego_v2.py — Option A : ParEGO corrigé + validation du front.

Corrige les trois défauts de l'optimisation v10 (cellule 48, traces/opt10) :
  1. Fenêtre : WIN = (1500, 24000) excluait la fuite de l'AddRoundKey(0) des octets 0 à 11 (t = 552 + 86 b).
     -> WIN_V2 = (300, 24000) : la métrique voit tout ce que voit l'attaquant.
  2. Espace : masques 0x003 / 0x007 seulement, grille de délais sans 536 ; les niveaux réels (m403) et la
     ligne de base « délais avant » n'en faisaient pas partie.
     -> MASKS_V2 = (0x003, 0x007, 0x403, 0x7FF), délai 536 ajouté, niveaux actuels injectés comme points de départ.
  3. Mesure : 3000 traces par point (résolution rho ≈ 0,077, saturation vers log10 N ≈ 3,7), une seule campagne.
     -> 6000 traces par point (résolution ≈ 0,055) et répétition des points du front.
Et ajoute :
  4. Une validation indépendante des points du front : 20 000 traces, attaque réelle (verify_n.success_curve)
     avec une sélection d'attaquant par MARGE DE DISTINCTION (rang puis écart à la meilleure mauvaise clé) au lieu
     du rho maximal de la bonne clé (qui choisissait des « points fantômes », ex. octets 11-13 de delais_avant_20k).
  5. Le tau de Kendall entre la métrique rapide (ParEGO) et l'effort mesuré : fidélité de la métrique.

Les fonctions lourdes sont parallélisées par octet (fork). Placer ce fichier dans projets2/.
Références : Knowles, IEEE TEVC 2006 (ParEGO) ; Mangard, CT-RSA 2004 (N(rho)) ;
Standaert, Malkin, Yung, EUROCRYPT 2009 (taux de succès) ; Clavier et al., CHES 2000 (intégration).
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

import noise_opt as no
import sca_metrics as sm
import verify_n as vn

WIN_V2 = (300, 24000)
MASKS_V2 = (0x003, 0x007, 0x403, 0x7FF)
PAIRS_V2 = ((0, 1), (1, 1), (1, 3), (3, 1), (3, 3), (4, 1), (4, 3))   # mêmes paires que la v10 (jitter / permutation)
RAW_GRID_V2 = {1: (10, 25, 50, 100, 200, 400, 536, 800), 2: no.RAW_GRID[2], 3: (0, 1, 2, 4), 4: no.RAW_GRID[4]}
BYTES = tuple(range(16))
MAX_SHIFT = 3000
ALPHA = 0.9999
NS_VAL = (20, 50, 100, 200, 500, 1000, 2000, 4000, 6000)
CFG_DELAIS_AVANT = no.Config(((1, 1), (4, 1)), ((1, 536),), 0x403)     # = traces/comp20k/delais_avant_20k


FM_SHIFT = 2                     # firmware v10 : délais à moyenne flottante, b = a >> 2 (commande 'J')


def _setup_v10(cfg):
    """cfg.setup() puis réglage explicite de la moyenne flottante : un réglage laissé par une campagne précédente
    (ex. fm_shift = 0) ne doit pas se propager aux mesures de ParEGO."""
    base = cfg.setup()

    def _s(nt):
        base(nt)
        import monitor_collect as mc
        mc._cmd(nt.target, "J", bytearray([FM_SHIFT]))
    return _s


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _safe(name):
    return name.replace("@", "_").replace(":", "").replace("+", "__").replace("|", "_")


def _nproc(nproc):
    """Par défaut : limité par la mémoire disponible (~3 Go par processus), voir banc_v10.nproc_sur."""
    if nproc:
        return int(nproc)
    try:
        import banc_v10
        return banc_v10.nproc_sur()
    except ImportError:
        return max(1, min((os.cpu_count() or 2) - 1, 4))


def _one_thread():
    """Évite la sur-souscription BLAS dans les processus fils (sans effet si threadpoolctl est absent)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# 1. Espace de recherche et référence propre
# --------------------------------------------------------------------------
def build_space(max_comps=4, masks=MASKS_V2, pairs=PAIRS_V2, raw_grid=RAW_GRID_V2):
    return no.candidate_space(max_comps=max_comps, masks=masks, pairs=pairs, raw_grid=raw_grid)


def make_clean(path_clean, window=WIN_V2):
    """Référence propre (même build, sans bruit) pour A3/A4, calculée sur la NOUVELLE fenêtre.
    Vérifie que les pics de l'AddRoundKey(0) (modèle 'xor', t < 1500) sont bien dans la référence."""
    cl = sm.clean_reference(path_clean, BYTES, window=window)
    pk = cl.get("views", {}).get("xor", {}).get("peaks", {})
    early = sorted(b for b, t in pk.items() if t < 1500)
    log("référence propre %s, fenêtre %s : %d octets ont leur pic 'xor' avant t=1500 (AddRoundKey(0)) %s"
        % (path_clean, window, len(early), early))
    return cl


# --------------------------------------------------------------------------
# 2. Métrique rapide (pire cas A1-A4 x modèles sbox/xor), parallélisée par octet
# --------------------------------------------------------------------------
_G = {}


def _eval_byte(b):
    _one_thread()
    g = _G
    r = sm.evaluate_rho_campaign(g["path"], bytes_=(b,), window=g["window"], alpha=g["alpha"],
                                 baseline_cycles=g["base"], clean=g["clean"], max_shift=g["ms"], use_n=g["use_n"])
    return b, r


def evaluate_parallel(path, clean, baseline_cycles, window=WIN_V2, alpha=ALPHA, max_shift=MAX_SHIFT,
                      nproc=None, use_n=None):
    """Même résultat que sm.evaluate_rho_campaign(path, bytes_=0..15, ...) mais un processus par octet.
    log10_n_est = N de Mangard de l'octet le plus faible (pire cas pour le défenseur)."""
    _G.update(path=path, clean=clean, base=baseline_cycles, window=tuple(window), alpha=alpha, ms=max_shift,
              use_n=use_n)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(_nproc(nproc), len(BYTES)), mp_context=mp.get_context("fork")) as ex:
        res = dict(ex.map(_eval_byte, BYTES))
    per = {b: res[b]["per_byte"][b] for b in res}
    wb = max(per, key=lambda b: per[b]["rho"])
    r0 = res[wb]
    st = dict(rho_star=float(per[wb]["rho"]), worst_byte=int(wb), resolution=float(r0["resolution"]),
              censored=bool(all(per[b]["censored"] for b in per)), n=int(r0["n"]), per_byte=per, path=path,
              overhead_pct=float(r0["overhead_pct"]), cycles_mean=float(r0["cycles_mean"]), window=list(window))
    sm.summarize_strength(st, alpha)
    w = sorted({res[b]["warning"] for b in res if "warning" in res[b]})
    if w:
        st["warning"] = w[0]
        log("ATTENTION", path, w[0])
    return st


def _summary_row(r):
    wb = r["per_byte"][r["worst_byte"]]
    return dict(log10_n=float(r["log10_n_est"]), overhead_pct=float(r["overhead_pct"]), rho_star=float(r["rho_star"]),
                resolution=float(r["resolution"]), censored=bool(r["censored"]), worst_byte=int(r["worst_byte"]),
                attacker=str(wb.get("attacker", "")), model=str(wb.get("model", "")), t=str(wb.get("t", "")),
                path=r["path"], proxy="v2 win=%d-%d" % tuple(r["window"]))


def bench_evaluator_v2(nt, key, clean, baseline_cycles, n_traces=6000, outdir="traces/opt11", seed=42,
                       nproc=None, alpha=ALPHA):
    """f(cfg) -> dict(log10_n, overhead_pct, ...) : capture réelle puis métrique v2. Même interface que
    noise_opt.bench_evaluator (utilisable par noise_opt.ParEGO)."""
    import sca_capture as sc

    def _f(cfg):
        path = sc.capture_campaign(nt, _safe(cfg.name), n_traces, key, seed=seed, outdir=outdir, setup=_setup_v10(cfg),
                                   meta=dict(config=cfg.to_dict(), proxy="v2"))
        return _summary_row(evaluate_parallel(path, clean, baseline_cycles, nproc=nproc, alpha=alpha))
    return _f


# --------------------------------------------------------------------------
# 3. Recalcul HORS BANC de l'historique v10 avec la métrique v2 (aucune capture)
# --------------------------------------------------------------------------
def rescore_history(old_path="traces/opt10/history.json", clean=None, baseline_cycles=None,
                    out_path="traces/opt10/history_v2.json", nproc=None, base_dir=None):
    """Réévalue chaque campagne de l'historique v10 avec la fenêtre v2. Le surcoût mesuré à l'époque est conservé
    (il ne dépend pas de la fenêtre). Reprise automatique si out_path existe déjà.
    base_dir : dossier de départ des chemins relatifs de l'historique (ex. /home/khadim/chipwhisperer/projets2)."""
    old = json.load(open(old_path))
    if base_dir:
        old = [dict(h, path=h["path"] if os.path.isabs(h["path"]) else os.path.join(base_dir, h["path"])) for h in old]
    new = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {h["config"]["name"] for h in new}
    for i, h in enumerate(old):
        name = h["config"]["name"]
        if name in done:
            continue
        if not os.path.isdir(h["path"]):
            log("absent, ignoré :", h["path"])
            continue
        t0 = time.time()
        r = _summary_row(evaluate_parallel(h["path"], clean, baseline_cycles, nproc=nproc))
        e = dict(h)
        e.update({k: v for k, v in r.items() if k != "overhead_pct"})
        e.update(log10_n_v1=h["log10_n"], attacker_v1=h.get("attacker"), how=h.get("how", "") + " | rescored v2")
        new.append(e)
        with open(out_path, "w") as f:
            json.dump(new, f, indent=1, default=float)
        log("[%2d/%d] %-70s v1=%.2f -> v2=%.2f (%s, octet %d, %s)  %.0f s" % (
            i + 1, len(old), name[:70], h["log10_n"], e["log10_n"], e["attacker"], e["worst_byte"], e["model"],
            time.time() - t0))
    return new


def parse_name(name):
    """'jitter@subbytes_pre:536+jitter@addroundkey_pre:536|m403' -> noise_opt.Config (meta.json des campagnes
    comp20k où `config` est une chaîne)."""
    body, m = name.split("|m")
    comps, raw = [], {}
    for part in body.split("+"):
        g, rest = part.split("@")
        z, r = rest.split(":")
        gi, zi = no.GENS.index(g), no.ZONES.index(z)
        comps.append((zi, gi)); raw[gi] = int(r)
    return no.Config(tuple(sorted(comps)), tuple(sorted(raw.items())), int(m, 16))


def config_of(path, levels_path="levels_3niv.json"):
    """Config d'une campagne. Les campagnes TresFort_* / Fort_* / Normal_* (sans champ `config` dans meta.json)
    sont rattachées aux niveaux de levels_3niv.json (index 2 / 1 / 0)."""
    c = json.load(open(os.path.join(path, "meta.json"))).get("config")
    if isinstance(c, dict):
        return no.Config.from_dict(c)
    if isinstance(c, str):
        return parse_name(c)
    base = os.path.basename(path.rstrip("/"))
    for prefix, i in (("TresFort", 2), ("Fort", 1), ("Normal", 0)):
        if base.startswith(prefix):
            return no.Config.from_dict(no.load_levels(levels_path)[i]["config"])
    raise ValueError("%s : pas de champ 'config' dans meta.json" % path)


def front(history, key="log10_n"):
    c = [h["overhead_pct"] for h in history]
    s = [h[key] for h in history]
    return [history[i] for i in sm.pareto_front(c, s)]


def compare_v1_v2(new):
    """Tableau v1 -> v2, tau de Kendall entre les deux métriques et points qui quittent / entrent dans le front."""
    from scipy.stats import kendalltau
    a = np.array([h["log10_n_v1"] for h in new]); b = np.array([h["log10_n"] for h in new])
    tau, p = kendalltau(a, b)
    f1 = {h["config"]["name"] for h in front(new, "log10_n_v1")}
    f2 = {h["config"]["name"] for h in front(new, "log10_n")}
    print("%-72s %7s %6s %6s %s" % ("config", "surcoût", "v1", "v2", "front"))
    for h in sorted(new, key=lambda h: h["overhead_pct"]):
        n = h["config"]["name"]
        tag = ("v1" if n in f1 else "  ") + ("+v2" if n in f2 else "   ")
        print("%-72s %+6.1f%% %6.2f %6.2f %s" % (n[:72], h["overhead_pct"], h["log10_n_v1"], h["log10_n"], tag))
    print("\ntau de Kendall (v1, v2) = %.2f  (p = %.3g) sur %d points" % (tau, p, len(new)))
    print("quittent le front :", sorted(f1 - f2) or "aucun")
    print("entrent dans le front :", sorted(f2 - f1) or "aucun")
    return dict(tau=float(tau), p=float(p), front_v1=sorted(f1), front_v2=sorted(f2))


# --------------------------------------------------------------------------
# 4. ParEGO v2 (même algorithme que noise_opt.ParEGO, entrées corrigées)
# --------------------------------------------------------------------------
def seed_configs(levels_path="levels_3niv.json", rescored=None, with_delais_avant=True):
    """Points de départ : niveaux actuels, ligne de base « délais avant », front v2 de l'historique recalculé
    (tel quel et en masque 0x403). Ces points sont MESURÉS (pas d'hypothèse) avant l'exploration."""
    cfgs = []
    if levels_path and os.path.exists(levels_path):
        cfgs += [no.Config.from_dict(L["config"]) for L in no.load_levels(levels_path) if L["config"]["comps"]]
    if with_delais_avant:
        cfgs.append(CFG_DELAIS_AVANT)
    for h in (front(rescored) if rescored else []):
        c = no.Config.from_dict(h["config"])
        cfgs += [c, no.Config(c.comps, c.raw, 0x403)]
    out, seen = [], set()
    for c in cfgs:
        if c.name not in seen:
            seen.add(c.name); out.append(c)
    return out


def run_parego_v2(nt, key, clean, baseline_cycles, seeds=(), history_path="traces/opt11/history.json",
                  n_init=6, n_iter=25, max_overhead=50, n_traces=6000, nproc=None, seed=0):
    """Reprend automatiquement là où il s'est arrêté (historique JSON)."""
    space = build_space()
    log("espace v2 : %d configurations" % len(space))
    ev = bench_evaluator_v2(nt, key, clean, baseline_cycles, n_traces=n_traces,
                            outdir=os.path.dirname(history_path) or ".", nproc=nproc)
    opt = no.ParEGO(space, ev, history_path=history_path, max_overhead=max_overhead, seed=seed)
    for c in seeds:
        if c.name not in opt._done():
            opt._record(c, ev(c), "seed")
    n_initial = sum(1 for h in opt.history if h["how"] == "initial")
    if n_initial < n_init:
        opt.initial_design(n_init - n_initial)
    n_done = sum(1 for h in opt.history if h["how"].startswith("parego"))
    opt.run(max(0, n_iter - n_done))
    opt.report()
    return opt


def replicate_front(nt, key, clean, baseline_cycles, history_path="traces/opt11/history.json", seed=43,
                    n_traces=6000, nproc=None):
    """Recapture les points du front avec d'AUTRES textes clairs (graine 43) et moyenne les deux mesures.
    Écrit history_moy.json (log10_n = moyenne, log10_n_sd = écart entre répétitions)."""
    hist = json.load(open(history_path))
    rep_path = history_path.replace(".json", "_rep.json")
    reps = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    ev = bench_evaluator_v2(nt, key, clean, baseline_cycles, n_traces=n_traces,
                            outdir=os.path.join(os.path.dirname(history_path) or ".", "rep"), seed=seed, nproc=nproc)
    for h in front(hist):
        n = h["config"]["name"]
        if n not in reps:
            reps[n] = ev(no.Config.from_dict(h["config"]))
            json.dump(reps, open(rep_path, "w"), indent=1, default=float)
            log("répétition %-60s %.2f -> %.2f" % (n[:60], h["log10_n"], reps[n]["log10_n"]))
    out = []
    for h in hist:
        e = dict(h)
        r = reps.get(h["config"]["name"])
        if r is not None:
            v = np.array([h["log10_n"], r["log10_n"]]); o = np.array([h["overhead_pct"], r["overhead_pct"]])
            e.update(log10_n=float(v.mean()), log10_n_sd=float(abs(v[0] - v[1]) / np.sqrt(2)),
                     overhead_pct=float(o.mean()), n_rep=2)
        out.append(e)
    moy = history_path.replace(".json", "_moy.json")
    json.dump(out, open(moy, "w"), indent=1, default=float)
    log("front après moyenne :", [h["config"]["name"] for h in front(out)])
    return out


# --------------------------------------------------------------------------
# 5. Validation : 20 000 traces, attaque réelle, sélection par marge de distinction
# --------------------------------------------------------------------------
def capture_validation(nt, key, configs, outdir="traces/val11", n_traces=20000, seed=0):
    """Graine 0 = mêmes textes clairs que traces/comp20k. Écrit outdir/index.json {nom de config: chemin}."""
    import sca_capture as sc
    os.makedirs(outdir, exist_ok=True)
    idx_path = os.path.join(outdir, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    for c in configs:
        idx[c.name] = sc.capture_campaign(nt, _safe(c.name), n_traces, key, seed=seed, outdir=outdir,
                                          setup=_setup_v10(c), meta=dict(config=c.to_dict()))
        json.dump(idx, open(idx_path, "w"), indent=1)
    return idx


def _split(N, seed=0):
    perm = np.random.default_rng(seed).permutation(N)                # même découpage que verify_n / sca_metrics
    return np.sort(perm[: N // 2]), np.sort(perm[N // 2:])


def _static_candidates(d, b, model, cl, h1, window, widths=(1, 4, 16, 64, 256), top=8, sep=20):
    """Les `top` meilleurs instants (rho de la bonne clé, moitié 1) pour chaque largeur : candidats A1/A2."""
    out = []
    for w in widths:
        if w >= window[1] - window[0]:
            continue
        r = np.abs(sm.corr_profile(d["traces"], d["pts"], d["key"], (b,), h1, window, w, model=model)[0])
        for _ in range(top):
            t = int(np.argmax(r))
            if r[t] <= 0:
                break
            out.append(dict(kind="static", byte=b, model=model, clean=cl, t=int(window[0] + t), w=int(w),
                            attacker="A1" if w == 1 else "A2(w=%d)" % w))
            r[max(0, t - sep):t + sep + w] = 0.0
    return out


def _margin(d, rec, H, h1, max_shift, k, z_sig=4.0):
    """Clé de tri (plus grand = meilleur). Un candidat n'est jugé sur (rang, marge) que si la corrélation de la
    bonne clé sur la moitié 1 est significative (> z_sig / sqrt(n1)) ; sinon il est classé après, par |rho|.
    Sans ce garde-fou, parmi ~100 candidats, l'un d'eux met la bonne clé en tête PAR HASARD (sur-apprentissage)."""
    f = vn.feature(d, rec, max_shift)
    s = vn._scores(H, f, h1)
    sc = s * (1.0 if s[k] >= 0 else -1.0)
    rank = int((sc > sc[k]).sum())
    mg = float(sc[k] - np.max(np.delete(sc, k)))
    sig = bool(abs(s[k]) > z_sig / np.sqrt(len(h1)))
    key = (1, -rank, mg) if sig else (0, 0, float(abs(s[k])))
    return key, rank, mg, f


def best_margin_byte(d, clean, b, max_shift=MAX_SHIFT, window=WIN_V2, seed=0, models=sm.MODELS, top=8,
                     extra=(), kinds=("static", "aligned", "shuffle")):
    """Candidats : A1/A2 (top-k instants x largeurs), A3 (réalignement), A4 (mélange), pour chaque modèle.
    Choix sur la moitié 1 par (rang de la bonne clé, puis marge sur la meilleure mauvaise clé), lecture de rho sur
    la moitié 2. Évite les points où la bonne clé corrèle mais où des clés voisines corrèlent autant (fantômes)."""
    N = len(d["traces"])
    h1, h2 = _split(N, seed)
    k = int(np.asarray(d["key"])[b])
    best = None
    for m in models:
        cl = sm._clean_view(clean, m)
        H = vn._hyp_all(d["pts"][:, b], m)
        cands = _static_candidates(d, b, m, cl, h1, window, top=top) if "static" in kinds else []
        sa = None if "aligned" not in kinds else sm.attack_strength_aligned(d["traces"], d["pts"], d["key"], cl, (b,), max_shift=max_shift, n=N,
                                        seed=seed, model=m)["per_byte"].get(b)
        if sa and sa.get("rho_sel", -1) >= 0:
            cands.append(dict(sa, kind="aligned", model=m, clean=cl))
        if "peaks_all" in cl and "shuffle" in kinds:
            ss = sm.attack_strength_shuffle(d["traces"], d["pts"], d["key"], cl, (b,), max_shift=max_shift, n=N,
                                            seed=seed, model=m)["per_byte"].get(b)
            if ss and ss.get("rho_sel", -1) >= 0:
                cands.append(dict(ss, kind="shuffle", model=m, clean=cl))
        cands += [dict(e) for e in extra if e.get("model") == m]
        for rec in cands:
            try:
                key, rank, mg, f = _margin(d, rec, H, h1, max_shift, k)
            except Exception as ex:                      # candidat hors trace, etc.
                log("candidat ignoré octet %d (%s) : %s" % (b, rec.get("kind"), ex))
                continue
            if best is None or key > best["_key"]:
                hk = H[:, k]
                rho2 = abs(np.corrcoef(hk[h2], f[h2])[0, 1])
                best = dict(rec, byte=b, rank_sel=rank, margin_sel=mg, rho=float(rho2), rho_sel=mg, _key=key,
                            significant=bool(key[0]), censored=bool(rho2 < 3.0 / np.sqrt(len(h2))))
    if best is not None:
        best.pop("_key", None)
    return best


def _best_byte(b):
    _one_thread()
    return b, best_margin_byte(_G["d"], _G["clean"], b, _G["ms"], _G["window"], _G["seed"])


def best_margin_parallel(d, clean, window=WIN_V2, max_shift=MAX_SHIFT, seed=0, nproc=None):
    _G.update(d=d, clean=clean, window=tuple(window), ms=max_shift, seed=seed)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(_nproc(nproc), len(BYTES)), mp_context=mp.get_context("fork")) as ex:
        r = dict(ex.map(_best_byte, BYTES))
    return {b: v for b, v in r.items() if v is not None}


def validate_campaign(path, clean, baseline_cycles, window=WIN_V2, ns=NS_VAL, trials=100, nproc=None, seed=0):
    """Effort restant (bits) et X_median (plus petit n tel que l'effort médian <= 2^32) avec sélection par marge,
    + métrique rapide v2 calculée sur la même campagne (pour le budget de fuite)."""
    d = sm._load(path)
    best = best_margin_parallel(d, clean, window, MAX_SHIFT, seed, nproc)
    out = vn.success_curve(path, clean, ns=ns, trials=trials, max_shift=MAX_SHIFT, window=window, best=best,
                           verbose=False)
    ns_ = list(out["ns"])
    med = [float(np.median(out["effort_bits"][n])) for n in ns_]
    proxy = evaluate_parallel(path, clean, baseline_cycles, window=window, nproc=nproc)
    res = dict(path=path, name=os.path.basename(path), win=list(window), ns=ns_, effort_med=med,
               effort_p10=[float(np.percentile(out["effort_bits"][n], 10)) for n in ns_],
               effort_p90=[float(np.percentile(out["effort_bits"][n], 90)) for n in ns_],
               octets=[float(out["mean_bytes"][n]) for n in ns_],
               X_median=next((n for n, e in zip(ns_, med) if e <= 32), None),
               overhead_pct=proxy["overhead_pct"], rho_star_20k=proxy["rho_star"],
               log10_n_20k=proxy["log10_n_est"], censored_20k=proxy["censored"],
               selection={int(b): dict(kind=r["kind"], model=r["model"], attacker=r.get("attacker", r["kind"]),
                                       t=int(r.get("t", -1)), w=int(r.get("w", 0)), rank_sel=r["rank_sel"],
                                       margin_sel=r["margin_sel"], significant=r.get("significant"), rho=r["rho"])
                          for b, r in best.items()})
    return res


def analyse_validation(history_path, val_json_paths, n_ref=2000):
    """Tau de Kendall entre la métrique rapide (historique ParEGO) et l'effort mesuré à n_ref traces ;
    front validé (surcoût, effort à n_ref)."""
    from scipy.stats import kendalltau
    hist = {h["config"]["name"]: h for h in json.load(open(history_path))}
    rows = []
    for p in val_json_paths:
        v = json.load(open(p))
        name = config_of(v["path"]).name
        e = v["effort_med"][v["ns"].index(n_ref)] if n_ref in v["ns"] else float("nan")
        rows.append(dict(name=name, overhead_pct=v["overhead_pct"], effort_bits=e, X_median=v["X_median"],
                         proxy=hist.get(name, {}).get("log10_n", float("nan")), log10_n_20k=v["log10_n_20k"]))
    ok = [r for r in rows if np.isfinite(r["proxy"]) and np.isfinite(r["effort_bits"])]
    tau, p = kendalltau([r["proxy"] for r in ok], [r["effort_bits"] for r in ok]) if len(ok) >= 3 else (np.nan, np.nan)
    P = [rows[i] for i in sm.pareto_front([r["overhead_pct"] for r in rows], [r["effort_bits"] for r in rows])]
    print("%-70s %7s %7s %10s %8s" % ("config", "surcoût", "proxy", "bits@%d" % n_ref, "X_med"))
    for r in sorted(rows, key=lambda r: r["overhead_pct"]):
        print("%-70s %+6.1f%% %7.2f %10.1f %8s %s" % (r["name"][:70], r["overhead_pct"], r["proxy"], r["effort_bits"],
                                                   r["X_median"] or ">%d" % NS_VAL[-1], "*" if r in P else ""))
    print("\ntau de Kendall (métrique rapide, effort à %d traces) = %.2f (p = %.3g, %d points)" % (n_ref, tau, p, len(ok)))
    return dict(rows=rows, front=[r["name"] for r in P], tau=float(tau), p=float(p))


def derive_levels_v2(val_json_paths, k=3, max_overhead=45, n_ref=2000, out="levels_v11.json"):
    """Niveaux de la politique choisis sur le front VALIDÉ (surcoût, effort à n_ref) ; rho du budget de fuite =
    rho* mesuré sur 20 000 traces (fenêtre v2). Format identique à noise_opt.save_levels (cellule 58 inchangée,
    mais l'arbre CART doit être réentraîné car leak_q16 change)."""
    rows = []
    for p in val_json_paths:
        v = json.load(open(p))
        if v["overhead_pct"] > max_overhead:
            continue
        cfg = config_of(v["path"]).to_dict()
        rows.append(dict(cfg=cfg, v=v, s=v["effort_med"][v["ns"].index(n_ref)]))
    P = [rows[i] for i in sm.pareto_front([r["v"]["overhead_pct"] for r in rows], [r["s"] for r in rows])]
    P.sort(key=lambda r: r["v"]["overhead_pct"])
    pick = [P[int(round(i))] for i in np.linspace(0, len(P) - 1, min(k, len(P)))]
    levels = []
    for r in pick:
        rho = float(r["v"]["rho_star_20k"])
        levels.append(dict(config=r["cfg"], overhead_pct=float(r["v"]["overhead_pct"]),
                           log10_n=float(r["v"]["log10_n_20k"]), rho=rho, leak=float(sm.leak_per_trace(rho)),
                           leak_q16=sm.leak_q16(rho), censored=bool(r["v"]["censored_20k"]),
                           effort_bits_ref=float(r["s"]), n_ref=n_ref, X_median=r["v"]["X_median"]))
    no.save_levels(levels, out)
    for i, L in enumerate(levels):
        print(i, L["config"]["name"][:70], "| %+.1f %%" % L["overhead_pct"], "| bits@%d %.1f" % (n_ref, L["effort_bits_ref"]),
              "| rho %.3f" % L["rho"])
    return levels
