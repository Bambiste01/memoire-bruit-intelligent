"""
sca_metrics.py — validation de l'attaquant réaligné (A2) et métriques honnêtes

Pourquoi ce module (constats sur l'étape 2) :
  * un rang obtenu après une attaque ÉCHOUÉE est uniforme sur 1..256 : le comparer d'une config à l'autre n'a
    aucun sens (bande du hasard : voir null_rank_band) ;
  * A2 n'a jamais été validé : sur la baseline, les lags valent 0 (test trivial). Il faut le tester avec des
    décalages CONNUS, injectés dans de vraies traces sans bruit (validate_aligner / scan_aligner).

v7 — métrique continue et attaquant A2 par intégration (section « Métriques v7 » en fin de fichier) :
  * attack_strength : corrélation de la bonne clé SANS biais de sélection (point et largeur choisis sur une
    moitié des traces, mesurés sur l'autre), pour A1 (échantillon unique) et A2 (intégration sur fenêtre
    glissante, Clavier-Coron-Dabbous CHES 2000) ;
  * n_from_rho : nombre de traces estimé (formule de Mangard, CT-RSA 2004 : N = 3 + 2 z² / atanh(rho)²) ;
  * calibrate_mangard : vérifie la formule contre le MTD mesuré sur la baseline (choix de alpha) ;
  * snr_profile / leak_map : carte de fuite par (ronde, zone) à partir des horodatages firmware ('T') ;
  * leak_q16 / leak_limit_q16 : constantes du budget de fuite embarqué.

Utilisation :
    import sca_capture as sc, sca_metrics as sm
    d = sc.load_campaign("traces/baseline_20k")
    sm.calibrate_mangard(d["traces"], d["pts"], d["key"])
    r = sm.evaluate_rho_campaign("traces/jit_rs_50", baseline_cycles=95783)
"""
from statistics import NormalDist

import numpy as np

import sca_eval as se
import acceleration;
acceleration.install()


# --------------------------------------------------------------------------
# Hasard et conversions
# --------------------------------------------------------------------------
def null_rank_band(n_bytes=4, trials=400000, seed=0, q=(5, 50, 95)):
    """Percentiles du rang médian de `n_bytes` octets quand la clé n'est PAS récupérée (rang ~ uniforme 1..256)."""
    r = np.random.default_rng(seed).integers(1, 257, (trials, n_bytes))
    return np.percentile(np.median(r, axis=1), q)


def mtd_from_rho(rho, alpha=0.99):
    """Nombre de traces approximatif pour distinguer la bonne clé avec confiance alpha, à partir de la corrélation
    rho de la bonne clé au point de fuite : n ~ 3 + 8 * (z_alpha / ln((1+rho)/(1-rho)))^2.
    Approximation classique (Mangard, Oswald, Popp, 'Power Analysis Attacks') — à vérifier dans l'ouvrage ;
    utile comme ORDRE DE GRANDEUR continu, pas comme MTD mesuré."""
    rho = np.clip(np.abs(np.asarray(rho, dtype=float)), 1e-6, 0.999999)
    z = NormalDist().inv_cdf(alpha)
    return 3.0 + 8.0 * (z / np.log((1.0 + rho) / (1.0 - rho))) ** 2


def rho_true_at(traces, pts, key, byte, cols, chunk=2000):
    """Corrélation de la bonne clé en des échantillons FIXES `cols` (pas d'argmax : aucun biais de sélection)."""
    cols = np.asarray(cols, dtype=int)
    N = len(traces)
    h = se.HW[se.SBOX[np.asarray(pts)[:, byte] ^ int(key[byte])]]
    sh, sh2 = h.sum(), (h ** 2).sum()
    st = np.zeros(len(cols)); st2 = np.zeros(len(cols)); sht = np.zeros(len(cols))
    for s in range(0, N, chunk):
        X = np.asarray(traces[s:s + chunk][:, cols], dtype=np.float64)
        st += X.sum(0); st2 += (X ** 2).sum(0); sht += h[s:s + chunk] @ X
    cov = N * sht - sh * st
    den = np.sqrt(max(N * sh2 - sh ** 2, 1e-12) * np.maximum(N * st2 - st ** 2, 1e-12))
    return cov / den


# --------------------------------------------------------------------------
# Validation de l'attaquant réaligné avec décalages connus
# --------------------------------------------------------------------------
def shift_traces(X, shifts):
    """Retarde chaque trace n de shifts[n] échantillons (le contenu se décale vers la droite) ; bord répété."""
    X = np.asarray(X)
    N, T = X.shape
    idx = np.clip(np.arange(T)[None, :] - np.asarray(shifts)[:, None], 0, T - 1)
    return X[np.arange(N)[:, None], idx]


def validate_aligner(traces, J, ref_start, ref_len, max_shift=None, n=600, window=None, seed=0, ref="clean"):
    """Injecte des décalages uniformes connus dans [0, J] sur n traces SANS bruit, réaligne avec se.resync, compare.

    ref = "clean" : motif de référence = moyenne des traces non décalées (attaquant qui dispose d'un exemplaire
                    propre du dispositif, cas réaliste) ; "self" : médiane des traces décalées (cas par défaut de resync).
    Coordonnées absolues (échantillons de la trace complète) ; `window=(a,b)` recadre pour économiser la mémoire.
    Succès : 95 % des erreurs de décalage <= 2 échantillons.
    """
    a, b = window if window is not None else (0, traces.shape[1])
    X = np.asarray(traces[:n, a:b], dtype=np.float32)
    rs = ref_start - a
    if max_shift is None:
        max_shift = J + 20
    rng = np.random.default_rng(seed)
    s = rng.integers(0, J + 1, len(X))
    Xs = shift_traces(X, s)
    refv = X[:, rs:rs + ref_len].mean(axis=0) if ref == "clean" else None
    try:
        _, lags = se.resync(Xs, ref_start=rs, ref_len=ref_len, max_shift=max_shift, ref=refv)
    except ValueError as e:
        return dict(J=J, ref_start=ref_start, ref_len=ref_len, max_shift=max_shift, ref=ref, error=str(e), ok=False)
    err = lags - s
    err = err - np.median(err)                                   # un décalage constant est sans importance
    p95 = float(np.percentile(np.abs(err), 95))
    return dict(J=J, ref_start=ref_start, ref_len=ref_len, max_shift=max_shift, ref=ref,
                err_std=float(err.std()), err_p95=p95, lag_min=int(lags.min()), lag_max=int(lags.max()),
                at_limit=bool(np.any(np.abs(lags) >= max_shift)), ok=bool(p95 <= 2.0))


def scan_aligner(traces, J_list=(5, 30, 100, 250, 800), ref_starts=(3900, 4100, 4260), ref_lens=(80, 160),
                 window=(2500, 11000), n=400, refs=("clean", "self"), verbose=True):
    """Grille de paramètres : quels (ref_start, ref_len) retrouvent les décalages connus pour chaque J ?"""
    rows = []
    for J in J_list:
        for rsx in ref_starts:
            for rl in ref_lens:
                for rf in refs:
                    r = validate_aligner(traces, J, rsx, rl, n=n, window=window, ref=rf)
                    rows.append(r)
    if verbose:
        print("%-5s %-9s %-8s %-6s | %-8s %-8s %-12s %s" % ("J", "ref_start", "ref_len", "ref", "err_std", "err_p95", "lags", "verdict"))
        for r in rows:
            if "error" in r:
                print("%-5d %-9d %-8d %-6s | %s" % (r["J"], r["ref_start"], r["ref_len"], r["ref"], r["error"][:60]))
                continue
            print("%-5d %-9d %-8d %-6s | %-8.1f %-8.1f %-12s %s%s" % (
                r["J"], r["ref_start"], r["ref_len"], r["ref"], r["err_std"], r["err_p95"],
                "%d/%d" % (r["lag_min"], r["lag_max"]), "OK" if r["ok"] else "ECHEC",
                "  (lag au bord)" if r["at_limit"] else ""))
        good = {}
        for r in rows:
            if r.get("ok"):
                good.setdefault((r["ref_start"], r["ref_len"], r["ref"]), []).append(r["J"])
        full = [k for k, v in good.items() if sorted(v) == sorted(J_list)]
        print("\nRéglages valides pour TOUS les J testés :", full if full else "aucun -> A2 non utilisable en l'état")
    return rows

# ==========================================================================
# Extensions : variance_profile, jitter_onset, scan_real_alignment
# ==========================================================================

def variance_profile(cfg_traces, ref_traces, window=(2500, 6000), n=1000):
    """
    Ratio var(cfg)/var(ref) par échantillon.
    Retourne (samples, ratio).
    """
    a, b = window
    C = np.asarray(cfg_traces[:n, a:b], dtype=np.float64)
    R = np.asarray(ref_traces[:n, a:b], dtype=np.float64)
    vc = C.var(axis=0)
    vr = R.var(axis=0) + 1e-12
    ratio = vc / vr
    samples = np.arange(a, b)
    return samples, ratio

def jitter_onset(samples, ratio, thr=2.0, min_len=20, start=None):
    """
    Premier échantillon où ratio > thr pendant au moins min_len échantillons consécutifs.
    """
    s0 = samples[0] if start is None else start
    mask = (samples >= s0) & (ratio > thr)
    run = 0
    for i, m in enumerate(mask):
        if samples[i] < s0:
            continue
        if m:
            run += 1
            if run >= min_len:
                return int(samples[i - min_len + 1])
        else:
            run = 0
    return None

def scan_real_alignment(cfg_traces, ref_traces, window=(2500, 11000),
                        control=(4300, 5300),
                        ref_starts=(3000, 3400, 3800, 4000, 4200, 4300),
                        ref_lens=(100, 200), max_shifts=(150, 400),
                        n=800):
    """
    Qualité d'alignement = std inter-traces (région contrôle) après resync
    divisée par std des traces propres sur la même région.
    ≈ 1.0 = bien aligné ; >> 1 = mauvais alignement.
    """
    import sca_eval as se
    a, b = window
    C = np.asarray(cfg_traces[:n, a:b], dtype=np.float32)
    R = np.asarray(ref_traces[:n, a:b], dtype=np.float32)
    c0, c1 = control[0] - a, control[1] - a
    # Dispersion de référence (traces propres)
    q0 = float(R[:, c0:c1].std(axis=0).mean())
    rows = []
    print(f"{'ref_start':>10} {'ref_len':>8} {'max_sh':>7} | {'qualité':>8}  verdict")
    for rs in ref_starts:
        for rl in ref_lens:
            for ms in max_shifts:
                rs_rel = rs - a
                if rs_rel - ms < 0 or rs_rel + rl + ms > C.shape[1]:
                    continue
                try:
                    # référence PROPRE
                    refv = R[:, rs_rel:rs_rel+rl].mean(axis=0)
                    al, lags = se.resync(C, ref_start=rs_rel, ref_len=rl,
                                         max_shift=ms, ref=refv)
                    q = float(al[:, c0:c1].std(axis=0).mean()) / (q0 + 1e-12)
                    ok = q <= 1.5
                    rows.append(dict(ref_start=rs, ref_len=rl, max_shift=ms,
                                     quality=q, lag_std=float(lags.std()),
                                     lag_min=int(lags.min()), lag_max=int(lags.max()),
                                     ok=ok))
                    print(f"{rs:10d} {rl:8d} {ms:7d} | {q:8.2f}  "
                          f"{'OK' if ok else 'ECHEC'}")
                except Exception as e:
                    print(f"{rs:10d} {rl:8d} {ms:7d} | erreur: {e}")
    good = [r for r in rows if r["ok"]]
    print(f"\nRéglages OK (qualité ≤ 1.5) : {len(good)} / {len(rows)}")
    if good:
        best = min(good, key=lambda r: r["quality"])
        print(f"Meilleur : ref_start={best['ref_start']}  ref_len={best['ref_len']}  "
              f"max_shift={best['max_shift']}  qualité={best['quality']:.2f}")
    return rows, q0


# ==========================================================================
# Métriques v7 : corrélation sans biais, A2 par intégration, N estimé, carte de fuite, budget de fuite
# ==========================================================================
import json as _json
import os as _os


def _z(alpha):
    return NormalDist().inv_cdf(alpha)


def n_from_rho(rho, alpha=0.9999):
    """Formule de Mangard (CT-RSA 2004) : N = 3 + 8 z² / ln²((1+rho)/(1-rho)) = 3 + 2 z² / atanh(rho)².
    Même formule que mtd_from_rho, alpha par défaut 0,9999 (à confirmer par calibrate_mangard)."""
    return mtd_from_rho(rho, alpha)


def leak_per_trace(rho):
    """Information apportée par une trace à l'attaquant : atanh(rho)². L'attaque réussit (au sens de Mangard)
    quand la somme sur les traces atteint 2 z² ; c'est la grandeur cumulée par le budget de fuite embarqué.
    Additivité exacte pour des traces de même rho, APPROXIMATION pour un mélange de niveaux (à dire dans le mémoire)."""
    rho = np.clip(np.abs(np.asarray(rho, dtype=float)), 0.0, 0.999999)
    return np.arctanh(rho) ** 2


def leak_q16(rho):
    """Valeur `leak_q16` d'un niveau dans noise_policy.h."""
    return int(round(float(leak_per_trace(rho)) * 65536))


def leak_limit_q16(alpha=0.9999, margin=1.0):
    """Limite du budget de fuite par clé : 2 z² / margin (margin > 1 = marge contre un attaquant meilleur
    que ceux évalués). Exemple : margin=10 -> la clé est déclarée épuisée à 10 % de l'effort de l'attaque."""
    return int(round(2.0 * _z(alpha) ** 2 / float(margin) * 65536))


def integrate(X, w):
    """Somme glissante sur w échantillons (attaque par intégration, Clavier et al. CHES 2000).
    Sortie [N, T-w+1] ; la colonne t couvre les échantillons t .. t+w-1."""
    X = np.asarray(X, dtype=np.float64)
    if w <= 1:
        return X
    c = np.cumsum(X, axis=1)
    c = np.concatenate([np.zeros((len(X), 1)), c], axis=1)
    return c[:, w:] - c[:, :-w]


MODELS = ("sbox", "xor")


def _hyp(pts_col, key_byte, model="sbox"):
    """Hypothèse de fuite (poids de Hamming) de la BONNE clé pour un octet.
    model="sbox" : HW(S(pt ^ k))  — sortie de la S-box (cible classique de la CPA, Brier et al. CHES 2004) ;
    model="xor"  : HW(pt ^ k)     — entrée de la S-box (sortie de l'AddRoundKey de la ronde 0)."""
    x = np.asarray(pts_col).astype(np.uint8) ^ np.uint8(key_byte)
    if model == "sbox":
        return se.HW[se.SBOX[x]]
    if model == "xor":
        return se.HW[x]
    raise ValueError("model inconnu : %r (attendu : %s)" % (model, MODELS))


def _true_hyp(pts, key, bytes_, model="sbox"):
    key = np.asarray(key, dtype=np.uint8)
    return np.stack([_hyp(np.asarray(pts)[:, b], key[b], model) for b in bytes_])     # [nb, N]


def corr_profile(traces, pts, key, bytes_, idx, window, w=1, chunk=500, model="sbox"):
    """rho[b, t] de la BONNE clé sur les traces `idx`, fenêtre (a, b), après intégration de largeur w."""
    a, b_ = window
    idx = np.sort(np.asarray(idx))
    H = _true_hyp(pts, key, bytes_, model)
    T = (b_ - a) - max(w, 1) + 1
    nb = len(bytes_)
    sh = np.zeros(nb); sh2 = np.zeros(nb)
    st = np.zeros(T); st2 = np.zeros(T); sht = np.zeros((nb, T))
    n = 0
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        X = integrate(np.asarray(traces[ii, a:b_], dtype=np.float64), w)
        Hc = H[:, ii]
        sh += Hc.sum(1); sh2 += (Hc ** 2).sum(1)
        st += X.sum(0); st2 += (X ** 2).sum(0); sht += Hc @ X
        n += len(ii)
    cov = n * sht - sh[:, None] * st[None, :]
    den = np.sqrt(np.maximum((n * sh2 - sh ** 2)[:, None], 1e-12) * np.maximum((n * st2 - st ** 2)[None, :], 1e-12))
    return cov / den


def attack_strength(traces, pts, key, bytes_=(0, 5, 10, 15), window=(3000, 11000),
                    widths=(1, 4, 16, 64, 256), n=None, seed=0, z_res=3.0, verbose=False, model="sbox"):
    """Corrélation de l'attaquant le plus fort parmi A1 (w=1) et A2 (intégration, w>1), SANS biais de sélection.
    model : "sbox" (sortie de S-box, défaut) ou "xor" (entrée de S-box).

    Protocole : les traces sont coupées en deux moitiés aléatoires. Sur la moitié 1, on choisit pour chaque
    octet la largeur w et l'échantillon t qui maximisent |rho| (ce que ferait l'attaquant) ; on lit |rho| en
    (w, t) sur la moitié 2. Un max sur des milliers d'échantillons serait biaisé vers le haut (~ sqrt(2 ln T / N))
    — d'où la séparation.

    Résolution : sous z_res / sqrt(n2), la valeur n'est pas distinguable de 0 : `censored=True` et N_est devient
    une BORNE INFÉRIEURE (il faut alors plus de traces, pas conclure à une sécurité infinie).
    """
    N = len(traces) if n is None else min(n, len(traces))
    perm = np.random.default_rng(seed).permutation(N)
    h1, h2 = perm[: N // 2], perm[N // 2:]
    res_thr = z_res / np.sqrt(len(h2))
    per = {b: dict(byte=b) for b in bytes_}
    for w in widths:
        if w >= window[1] - window[0]:
            continue
        p1 = corr_profile(traces, pts, key, bytes_, h1, window, w, model=model)
        p2 = corr_profile(traces, pts, key, bytes_, h2, window, w, model=model)
        for i, b in enumerate(bytes_):
            t = int(np.argmax(np.abs(p1[i])))
            s1, s2 = float(abs(p1[i, t])), float(abs(p2[i, t]))
            rec = per[b]
            if w == 1:
                rec.update(rho_a1=s2, t_a1=window[0] + t)
            if s1 > rec.get("_s1", -1):
                rec.update(_s1=s1, rho=s2, w=int(w), t=window[0] + t)
            if verbose:
                print("octet %2d  w=%4d  t=%6d  |rho| sélection=%.4f  test=%.4f" % (b, w, window[0] + t, s1, s2))
    for rec in per.values():
        rec["rho_sel"] = rec.pop("_s1")                  # score de sélection (moitié 1), sert à comparer les attaquants
        rec["attacker"] = "A1" if rec["w"] == 1 else "A2(w=%d)" % rec["w"]
        rec["censored"] = bool(rec["rho"] < res_thr)
    worst = max(per.values(), key=lambda r: r["rho"])
    return dict(per_byte=per, n=int(N), resolution=float(res_thr),
                rho_star=float(worst["rho"]), worst_byte=int(worst["byte"]),
                censored=bool(all(r["censored"] for r in per.values())))


def summarize_strength(st, alpha=0.9999):
    """Ajoute N estimé (Mangard) ; si censuré, N_est = borne inférieure calculée à la résolution."""
    rho = max(st["rho_star"], st["resolution"]) if st["censored"] else st["rho_star"]
    st["n_est"] = float(n_from_rho(rho, alpha))
    st["n_est_is_lower_bound"] = bool(st["censored"])
    st["log10_n_est"] = float(np.log10(st["n_est"]))
    st["leak_q16"] = leak_q16(rho)
    return st


def calibrate_mangard(traces, pts, key, bytes_=(0, 5, 10, 15), crop=300, n_max=None, n_perm=20,
                      alphas=(0.99, 0.999, 0.9999)):
    """Compare le MTD MESURÉ (GE/SR par ré-échantillonnage, sca_eval) au N prédit par la formule de Mangard
    sur des traces SANS bruit. Choisir l'alpha dont le rapport mesuré/prédit est le plus proche de 1, et
    l'utiliser partout ensuite (à documenter dans le mémoire)."""
    N = len(traces)
    rho_full = se.corr_all_bytes(traces, pts, key)
    rows = []
    for b in bytes_:
        t = int(np.argmax(np.abs(rho_full[b])))
        a_, b_ = max(0, t - crop), min(traces.shape[1], t + crop)
        X = np.asarray(traces[:, a_:b_], dtype=np.float32)
        r = se.ge_sr_mtd(X, pts, key, byte=b, n_perm=n_perm,
                         checkpoints=se.default_checkpoints(n_max or N // 2, n_min=2, count=30))
        rho = float(abs(rho_full[b, t]))
        row = dict(byte=b, rho=rho, mtd=r["mtd_median"])
        for al in alphas:
            row["pred_%g" % al] = float(n_from_rho(rho, al))
        rows.append(row)
    print("%-5s %-7s %-8s " % ("octet", "|rho|", "MTD mes.") + " ".join("N(a=%g)" % a for a in alphas))
    for r in rows:
        print("%-5d %-7.3f %-8s " % (r["byte"], r["rho"], ("%g" % r["mtd"]) if np.isfinite(r["mtd"]) else "inf")
              + " ".join("%-9.1f" % r["pred_%g" % a] for a in alphas))
    for a in alphas:
        ratios = [r["mtd"] / r["pred_%g" % a] for r in rows if np.isfinite(r["mtd"])]
        if ratios:
            print("alpha=%g : rapport mesuré/prédit médian = %.2f" % (a, np.median(ratios)))
    return rows


# --------------------------------------------------------------------------
# Évaluation d'une campagne et comparaison
# --------------------------------------------------------------------------
def _load(path):
    d = dict(traces=np.load(_os.path.join(path, "traces.npy"), mmap_mode="r"),
             pts=np.load(_os.path.join(path, "pts.npy")))
    with open(_os.path.join(path, "meta.json")) as f:
        d["meta"] = _json.load(f)
    d["key"] = np.frombuffer(bytes.fromhex(d["meta"]["key"]), dtype=np.uint8)
    tp = _os.path.join(path, "trig.npy")
    d["trig"] = np.load(tp) if _os.path.exists(tp) else None
    return d


def _corr_all_bytes_model(traces, pts, key, model="sbox", chunk=1000):
    """rho[b, t] pour les 16 octets, hypothèse = _hyp(model) (clé correcte)."""
    N, T = traces.shape
    key = np.asarray(key, dtype=np.uint8)
    Hall = np.stack([_hyp(np.asarray(pts)[:N, b], key[b], model) for b in range(16)]).astype(np.float64)
    sh = Hall.sum(1); sh2 = (Hall ** 2).sum(1)
    st = np.zeros(T); st2 = np.zeros(T); sht = np.zeros((16, T))
    for s0 in range(0, N, chunk):
        X = np.asarray(traces[s0:s0 + chunk], dtype=np.float64)
        st += X.sum(0); st2 += (X ** 2).sum(0)
        sht += Hall[:, s0:s0 + chunk] @ X
    cov = N * sht - sh[:, None] * st[None, :]
    den = np.sqrt(np.maximum((N * sh2 - sh ** 2)[:, None], 1e-12) * np.maximum((N * st2 - st ** 2)[None, :], 1e-12))
    return cov / den


def _peaks_from_rho(rho, a, b_, bytes_, max_points, min_rho, rel, sep):
    peaks_all = {b: a + int(np.argmax(np.abs(rho[b, a:b_]))) for b in range(16)}
    peaks_multi = {}
    for b in range(16):
        r = np.abs(rho[b, a:b_]).copy()
        top = float(r.max()); lst = []
        while len(lst) < max_points:
            t = int(np.argmax(r))
            if r[t] < max(min_rho, rel * top):
                break
            lst.append(a + t)
            r[max(0, t - sep):t + sep] = 0.0
        peaks_multi[b] = lst
    return dict(peaks={b: peaks_all[b] for b in bytes_}, peaks_all=peaks_all, peaks_multi=peaks_multi)


def clean_reference(path, bytes_=(0, 5, 10, 15), n=4000, window=None, max_points=4, min_rho=0.25, rel=0.3, sep=40,
                    models=MODELS):
    """Référence propre (SANS bruit, MÊME build) pour les attaquants A3/A4 : trace moyenne et positions des pics
    de fuite de chaque octet, pour chaque modèle de fuite (sortie S-box 'sbox', entrée S-box 'xor').
    clean[...] = vue 'sbox' (compatibilité) ; clean['views'][modèle] = vue de chaque modèle.
    Hypothèse : l'attaquant dispose d'un exemplaire sans contre-mesure (Kerckhoffs), défavorable au défenseur."""
    d = _load(path)
    N = min(n, len(d["traces"]))
    X = d["traces"][:N]
    mean = np.zeros(X.shape[1])
    for s0 in range(0, N, 1000):
        mean += np.asarray(X[s0:s0 + 1000], dtype=np.float64).sum(0)
    mean /= N
    a, b_ = window if window is not None else (0, X.shape[1])
    views = {}
    for m in models:
        rho = _corr_all_bytes_model(X, d["pts"][:N], d["key"], m)
        v = _peaks_from_rho(rho, a, b_, bytes_, max_points, min_rho, rel, sep)
        v.update(mean=mean, rho_profile=rho, path=path, model=m)
        views[m] = v
    out = dict(views[models[0]])
    out["views"] = views
    return out


def _clean_view(clean, model):
    """Vue de la référence propre pour un modèle donné (repli sur la vue de base si absente)."""
    if clean is None:
        return None
    v = clean.get("views", {}).get(model)
    return v if v is not None else clean


def attack_strength_aligned(traces, pts, key, clean, bytes_=(0, 5, 10, 15), max_shift=500, ref_len=60, gap=5,
                            widths=(1, 4, 16, 64, 256, 1024), post=2000, n=None, seed=0, z_res=3.0, model="sbox"):
    """Attaquant A3 : réalignement statique sur un MOTIF PROPRE situé juste avant le point de fuite de chaque
    octet (Mangard, Oswald, Popp 2007, ch. 8 ; contre les délais aléatoires : Clavier et al. CHES 2000,
    Coron & Kizhvatov CHES 2009). Un délai inséré AVANT ce motif est entièrement compensé : c'est la
    faiblesse connue d'un délai aléatoire unique. Le réalignement est tenté sur CHAQUE point où la variable
    fuit dans la trace propre (clean["peaks_multi"] : écriture de la S-box, puis relectures) et l'attaquant
    garde le meilleur. Même protocole sans biais que attack_strength.
    Largeurs w > 4 : après réalignement, intégration sur w échantillons (jusqu'à `post` après le point de fuite)
    = attaque intégrée contre le mélange et les délais ENTRE créneaux, sans hypothèse sur la géométrie des
    créneaux (le motif d'instructions, aligné, n'ajoute pas de variance ; seule la fuite des autres créneaux
    en ajoute) — libellé « A4(intég,w=...) »."""
    N = len(traces) if n is None else min(n, len(traces))
    T = traces.shape[1]
    perm = np.random.default_rng(seed).permutation(N)
    h1, h2 = perm[: N // 2], perm[N // 2:]
    res_thr = z_res / np.sqrt(len(h2))
    per = {}
    for b in bytes_:
        points = clean.get("peaks_multi", {}).get(b) or [int(clean["peaks"][b])]
        rec = dict(byte=b, rho_sel=-1.0)
        for k, p in enumerate(points):
            p = int(p)
            rs_abs = p - gap - ref_len                           # début du motif (coordonnées absolues)
            a = max(0, rs_abs - max_shift)
            e = min(T, p + max_shift + post)
            ms = min(max_shift, rs_abs - a, e - (rs_abs + ref_len) - 1)
            if ms < 1:
                continue
            X = np.asarray(traces[:N, a:e], dtype=np.float32)
            ref = np.asarray(clean["mean"][rs_abs:rs_abs + ref_len], dtype=np.float32)
            al, lags = se.resync(X, ref_start=rs_abs - a, ref_len=ref_len, max_shift=ms, ref=ref)
            for w in widths:
                if w >= al.shape[1]:
                    continue
                p1 = corr_profile(al, pts, key, (b,), h1, (0, al.shape[1]), w, model=model)[0]
                p2 = corr_profile(al, pts, key, (b,), h2, (0, al.shape[1]), w, model=model)[0]
                t = int(np.argmax(np.abs(p1)))
                s1, s2 = float(abs(p1[t])), float(abs(p2[t]))
                if s1 > rec["rho_sel"]:
                    lab = ("A3(align%s)" % ("" if w == 1 else ",w=%d" % w)) if w <= 4 \
                        else "A4(intég,w=%d)" % w
                    rec.update(rho_sel=s1, rho=s2, w=int(w), point=p, t=int(a + ms + t), lag_std=float(lags.std()),
                               lag_at_limit=float(np.mean(np.abs(lags) >= ms)), attacker=lab)
        if rec["rho_sel"] < 0:
            continue
        rec["censored"] = bool(rec["rho"] < res_thr)
        per[b] = rec
    return dict(per_byte=per, n=int(N), resolution=float(res_thr))


def shuffle_group(b):
    """Octets qui partagent les créneaux possibles de l'octet b sous ZONE_SUBBYTES_PRE (permutation de l'ordre
    des 4 itérations de la boucle externe de SubBytes dans aes.c) : même b // 4. Vérifié sur vos baselines :
    pic(b) = p0 + pas_externe x (b % 4) + pas_interne x (b // 4)."""
    return [4 * (b // 4) + k for k in range(4)]


def attack_strength_shuffle(traces, pts, key, clean, bytes_=(0, 5, 10, 15), max_shift=500, ref_len=60, gap=5,
                            widths=(1, 3, 5), n=None, seed=0, z_res=3.0, group=shuffle_group, model="sbox"):
    """Attaquant A4 contre le mélange (shuffling) : réalignement sur un motif propre précédant le premier
    créneau possible, puis SOMME des échantillons aux K créneaux candidats où l'octet peut être traité
    (attaque intégrée : Clavier et al. CHES 2000 ; Tillich & Herbst CT-RSA 2008 ; analyse du mélange :
    Veyrat-Charvillon et al. ASIACRYPT 2012). Théorie : rho_A4 ~ rho / sqrt(K), contre rho / K sans intégration."""
    if "peaks_all" not in clean:
        raise ValueError("clean_reference() récent requis (peaks_all)")
    N = len(traces) if n is None else min(n, len(traces))
    T = traces.shape[1]
    perm = np.random.default_rng(seed).permutation(N)
    h1, h2 = np.sort(perm[: N // 2]), np.sort(perm[N // 2:])
    res_thr = z_res / np.sqrt(len(h2))
    per = {}
    key = np.asarray(key, dtype=np.uint8)
    for b in bytes_:
        cand = sorted(int(clean["peaks_all"][c]) for c in group(b))
        rs_abs = cand[0] - gap - ref_len
        a = max(0, rs_abs - max_shift)
        e = min(T, cand[-1] + max_shift + 60)
        ms = min(max_shift, rs_abs - a, e - (rs_abs + ref_len) - 1)
        if ms < 1:
            continue
        X = np.asarray(traces[:N, a:e], dtype=np.float32)
        ref = np.asarray(clean["mean"][rs_abs:rs_abs + ref_len], dtype=np.float32)
        al, lags = se.resync(X, ref_start=rs_abs - a, ref_len=ref_len, max_shift=ms, ref=ref)
        h = _hyp(np.asarray(pts)[:N, b], key[b], model)
        rec = dict(byte=b, lag_std=float(lags.std()), lag_at_limit=float(np.mean(np.abs(lags) >= ms)), K=len(cand))
        for w in widths:
            S = np.zeros(N)
            for q in cand:
                j = q - a - ms
                lo, hi = max(0, j - w // 2), min(al.shape[1], j + w // 2 + 1)
                S += al[:, lo:hi].sum(1)
            r1 = abs(np.corrcoef(h[h1], S[h1])[0, 1]); r2 = abs(np.corrcoef(h[h2], S[h2])[0, 1])
            if r1 > rec.get("rho_sel", -1):
                rec.update(rho_sel=float(r1), rho=float(r2), w=int(w), attacker="A4(mélange,K=%d)" % len(cand))
        rec["censored"] = bool(rec["rho"] < res_thr)
        per[b] = rec
    return dict(per_byte=per, n=int(N), resolution=float(res_thr))


def _merge_attackers(st, st3, tag):
    """Pour chaque octet, garde l'attaquant au meilleur score de SÉLECTION (moitié 1), puis lit sa valeur de
    test (moitié 2) : le choix de l'attaquant n'introduit pas de biais."""
    for b, r3 in st3["per_byte"].items():
        rec = st["per_byte"][b]
        rec["rho_" + tag] = r3["rho"]
        rec[tag + "_lag_std"] = r3["lag_std"]
        rec[tag + "_lag_at_limit"] = r3["lag_at_limit"]
        if r3["rho_sel"] > rec["rho_sel"]:
            rec.update(rho=r3["rho"], rho_sel=r3["rho_sel"], w=r3["w"], attacker=r3["attacker"],
                       censored=r3["censored"], t=r3.get("t", ""))
    worst = max(st["per_byte"].values(), key=lambda r: r["rho"])
    st.update(rho_star=float(worst["rho"]), worst_byte=int(worst["byte"]),
              censored=bool(all(r["censored"] for r in st["per_byte"].values())))
    return st


def _merge_models(base, other, tag):
    """Par octet, garde le modèle dont l'attaquant a le meilleur score de SÉLECTION (moitié 1) ;
    la valeur lue est celle de la moitié 2 (pas de biais de sélection)."""
    for b, ro in other["per_byte"].items():
        rb = base["per_byte"][b]
        rb["rho_" + tag] = ro["rho"]
        if ro["rho_sel"] > rb["rho_sel"]:
            keep = {k: v for k, v in rb.items() if k.startswith("rho_") and k.endswith(("sbox", "xor"))}
            rb.clear(); rb.update(ro); rb.update(keep)
            rb["rho_" + tag] = ro["rho"]
            rb["model"] = tag
    worst = max(base["per_byte"].values(), key=lambda r: r["rho"])
    base.update(rho_star=float(worst["rho"]), worst_byte=int(worst["byte"]),
                censored=bool(all(r["censored"] for r in base["per_byte"].values())))
    return base


def _eval_one_model(d, clean, model, bytes_, window, widths, use_n, seed, max_shift):
    st = attack_strength(d["traces"], d["pts"], d["key"], bytes_, window, widths, n=use_n, seed=seed, model=model)
    N = st["n"]
    trig = d["trig"][:N] if d["trig"] is not None else None
    for r in st["per_byte"].values():
        r["model"] = model
    if clean is not None:
        ms = max_shift if max_shift is not None else (
            int(min(trig.max() - trig.min(), 3000)) + 50 if trig is not None else 500)
        st3 = attack_strength_aligned(d["traces"], d["pts"], d["key"], clean, bytes_, max_shift=ms, n=N,
                                      seed=seed, model=model)
        _merge_attackers(st, st3, "al")
        if "peaks_all" in clean:
            st4 = attack_strength_shuffle(d["traces"], d["pts"], d["key"], clean, bytes_, max_shift=ms, n=N,
                                          seed=seed, model=model)
            _merge_attackers(st, st4, "cr")
        st["a3_max_shift"] = ms
    for r in st["per_byte"].values():
        r["model"] = model
    return st


def evaluate_rho_campaign(path, bytes_=(0, 5, 10, 15), window=(3000, 11000), widths=(1, 4, 16, 64, 256),
                          use_n=None, alpha=0.9999, baseline_cycles=None, seed=0, clean=None, max_shift=None,
                          models=MODELS):
    """Évaluation v10 : surcoût + attaquant le plus fort parmi A1, A2, A3, A4 (si `clean`) ET parmi les modèles
    de fuite `models` (sortie S-box 'sbox', entrée S-box 'xor') + N estimé. Le meilleur (modèle, attaquant) est
    choisi par octet sur la moitié 1, lu sur la moitié 2. N = Mangard : pour le modèle 'xor', ambiguïté k / k^0xFF
    et clés voisines => N réel >= N de Mangard à faible rho (estimation pessimiste pour le défenseur)."""
    d = _load(path)
    st = None
    for m in models:
        sm_ = _eval_one_model(d, _clean_view(clean, m), m, bytes_, window, widths, use_n, seed, max_shift)
        st = sm_ if st is None else _merge_models(st, sm_, m)
        if st is sm_:
            for r in st["per_byte"].values():
                r["rho_" + m] = r["rho"]
    N = st["n"]
    trig = d["trig"][:N] if d["trig"] is not None else None
    summarize_strength(st, alpha)
    cyc = float(trig.mean()) if trig is not None else float("nan")
    st.update(path=path, name=_os.path.basename(path), cycles_mean=cyc,
              overhead_pct=(100.0 * (cyc / baseline_cycles - 1.0)) if baseline_cycles else float("nan"))
    if baseline_cycles and trig is not None and clean is not None:
        T = d["traces"].shape[1]
        delay_min = float(trig.min()) - baseline_cycles
        if delay_min > T - max(clean["peaks"].values()):
            st["warning"] = ("le délai minimal (%.0f cycles) peut repousser la fuite HORS de la capture : "
                             "évaluation non fiable" % delay_min)
    return st


def compare_rho(paths, baseline_cycles, **kw):
    rows = [evaluate_rho_campaign(p, baseline_cycles=baseline_cycles, **kw) for p in paths]
    print("%-24s %8s | %-7s %-5s %-19s %-6s %-6s | %-10s | %s" % ("config", "surcoût", "rho*", "modèl", "attaquant", "octet", "échant",
                                                       "N estimé", "rho A1 / réaligné (A3-A4) / créneaux 4 (octet faible)"))
    for r in rows:
        wb = r["per_byte"][r["worst_byte"]]
        extra = "%.3f / %s / %s" % (wb.get("rho_a1", float("nan")),
                                    ("%.3f" % wb["rho_al"]) if "rho_al" in wb else "-",
                                    ("%.3f" % wb["rho_cr"]) if "rho_cr" in wb else "-")
        print("%-24s %+7.1f%% | %-7.4f %-5s %-19s %-6d %-6s | %s%-9.0f | %s%s" % (
            r["name"], r["overhead_pct"], r["rho_star"], wb.get("model", "?"), wb["attacker"], r["worst_byte"], str(wb.get("t", "")),
            ">" if r["n_est_is_lower_bound"] else " ", r["n_est"], extra,
            ("   ATTENTION : " + r["warning"]) if "warning" in r else ""))
    return rows


def pareto_front(costs, securities):
    """Indices non dominés : coût minimal, sécurité (ex. log10 N_est) maximale."""
    c = np.asarray(costs, float); s = np.asarray(securities, float)
    keep = []
    for i in range(len(c)):
        dom = (c <= c[i]) & (s >= s[i]) & ((c < c[i]) | (s > s[i]))
        if not dom.any():
            keep.append(i)
    return sorted(keep, key=lambda i: c[i])


# --------------------------------------------------------------------------
# Carte de fuite : SNR par échantillon, puis par (ronde, zone)
# --------------------------------------------------------------------------
def snr_profile(traces, pts, key, byte, window=(3000, 11000), n=None, chunk=1000, model="value"):
    """SNR = Var(E[X | v]) / E[Var(X | v)] (Mangard, Oswald, Popp). v = sortie de S-box (256 classes)
    ou son poids de Hamming (model="hw", 9 classes, plus robuste avec peu de traces)."""
    a, b_ = window
    N = len(traces) if n is None else min(n, len(traces))
    v = se.SBOX[np.asarray(pts[:N, byte]) ^ int(key[byte])]
    if model == "hw":
        v = se.HW[v].astype(int)
    K = int(v.max()) + 1
    T = b_ - a
    cnt = np.zeros(K); s1 = np.zeros((K, T)); s2 = np.zeros((K, T))
    for s in range(0, N, chunk):
        X = np.asarray(traces[s:s + chunk, a:b_], dtype=np.float64)
        vv = v[s:s + chunk]
        np.add.at(cnt, vv, 1)
        np.add.at(s1, vv, X)
        np.add.at(s2, vv, X ** 2)
    ok = cnt > 1
    mu = s1[ok] / cnt[ok, None]
    var = s2[ok] / cnt[ok, None] - mu ** 2
    return np.var(mu, axis=0) / (np.mean(var, axis=0) + 1e-18)


ZONE_NAMES = ("round_start", "subbytes_pre", "subbytes_byte", "mixcolumns",
              "addroundkey_pre", "addroundkey_byte", "round_end")


def leak_map(profile, window_start, zone_ts, offset, samples_per_cycle=1.0, top=15):
    """Affecte à chaque (ronde, zone) le maximum de `profile` (SNR ou |rho|) entre son horodatage et la zone
    suivante. zone_ts : {(ronde, zone): cycles depuis le début du chiffrement} (sca_capture.read_zone_timestamps).
    offset : échantillon correspondant au cycle 0 (calibrate_ts_offset)."""
    ev = sorted(((c, k) for k, c in zone_ts.items() if c is not None), key=lambda x: x[0])
    rows = []
    for i, (c, k) in enumerate(ev):
        s0 = int(round(offset + c * samples_per_cycle)) - window_start
        s1 = (int(round(offset + ev[i + 1][0] * samples_per_cycle)) - window_start) if i + 1 < len(ev) else len(profile)
        s0, s1 = max(s0, 0), min(s1, len(profile))
        if s1 <= s0:
            continue
        j = s0 + int(np.argmax(profile[s0:s1]))
        rows.append(dict(round=k[0], zone=ZONE_NAMES[k[1]], start=s0 + window_start, end=s1 + window_start,
                         peak=float(profile[j]), peak_sample=j + window_start))
    rows.sort(key=lambda r: -r["peak"])
    for r in rows[:top]:
        print("ronde %2d  %-17s [%6d, %6d)  pic %.4f @ %d" % (r["round"], r["zone"], r["start"], r["end"],
                                                              r["peak"], r["peak_sample"]))
    return rows


def calibrate_ts_offset(zone_ts, peak_sample_byte0, samples_per_cycle=1.0):
    """Le premier appel SUBBYTES_BYTE de la ronde 1 suit immédiatement la S-box de l'octet 0 (sans permutation) :
    offset = pic CPA de l'octet 0 - horodatage. À contrôler visuellement sur la trace moyenne."""
    return float(peak_sample_byte0 - zone_ts[(1, 2)] * samples_per_cycle)