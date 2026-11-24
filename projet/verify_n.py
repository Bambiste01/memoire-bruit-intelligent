"""
verify_n.py — vérification DIRECTE du nombre de traces nécessaire (taux de succès de la CPA).

Pourquoi : sca_metrics donne N = formule de Mangard appliquée au rho mesuré (estimation). Ici on fait
vraiment l'attaque : pour n traces tirées au hasard, on classe les 256 valeurs possibles de chaque octet
de clé et on regarde si la BONNE valeur (connue de l'évaluateur, comme pour tout banc d'évaluation) est
première. Taux de succès (SR) de Standaert, Malkin, Yung (EUROCRYPT 2009), calculé par ré-échantillonnage.

Attaquant : le même que dans l'évaluateur (réalignement sur la référence propre, intégration, mélange),
choisi par octet sur la moitié 1 des traces ; les tirages ne se font QUE dans la moitié 2 (pas de biais de
sélection). Le choix du point/largeur utilise la bonne clé (attaquant « oracle ») : estimation prudente
pour le défenseur.

Limites à écrire dans le mémoire : n est limité à la taille de la moitié 2 (ex. 1500 pour 3000 traces) et
les tirages se recouvrent quand n s'en approche.
"""
import numpy as np

import sca_eval as se
import sca_metrics as sm
import acceleration;
acceleration.install()
DEFAULT_NS = (20, 50, 100, 200, 300, 500, 800, 1200)


# --------------------------------------------------------------------------
# Choix de l'attaquant par octet (même logique que l'évaluateur)
# --------------------------------------------------------------------------
def best_attackers(d, clean, bytes_, max_shift, models=sm.MODELS, seed=0, window=(1500, 24000),
                   widths=(1, 4, 16, 64, 256)):
    """Par octet : (modèle, attaquant) au meilleur score de SÉLECTION (moitié 1) parmi
    A1/A2 (échantillon fixe / intégré), A3 / A4-intégré (réalignement) et A4-mélange.
    Retourne {octet: dict(kind, model, clean, ...)}."""
    N = len(d["traces"])
    best = {}
    for m in models:
        cl = sm._clean_view(clean, m)
        s0 = sm.attack_strength(d["traces"], d["pts"], d["key"], bytes_, window, widths, n=N, seed=seed, model=m)
        for b, r in s0["per_byte"].items():
            if r["rho_sel"] > best.get(b, {}).get("rho_sel", -1.0):
                best[b] = dict(r, kind="static", model=m, clean=cl)
        sa = sm.attack_strength_aligned(d["traces"], d["pts"], d["key"], cl, bytes_, max_shift=max_shift,
                                        n=N, seed=seed, model=m)
        for b, r in sa["per_byte"].items():
            if r["rho_sel"] > best.get(b, {}).get("rho_sel", -1.0):
                best[b] = dict(r, kind="aligned", model=m, clean=cl)
        if "peaks_all" in cl:
            ss = sm.attack_strength_shuffle(d["traces"], d["pts"], d["key"], cl, bytes_, max_shift=max_shift,
                                            n=N, seed=seed, model=m)
            for b, r in ss["per_byte"].items():
                if r["rho_sel"] > best.get(b, {}).get("rho_sel", -1.0):
                    best[b] = dict(r, kind="shuffle", model=m, clean=cl)
    return best


def feature(d, rec, max_shift, gap=5, ref_len=60, post=2000):
    """Une valeur par trace (échantillon réaligné, éventuellement intégré) pour l'attaquant `rec`."""
    traces = d["traces"]
    N, T = traces.shape
    cl, b = rec["clean"], rec["byte"]
    if rec["kind"] == "static":
        t, w = int(rec["t"]), int(rec["w"])
        return np.asarray(traces[:, t:t + w], dtype=np.float64).sum(1)
    if rec["kind"] == "aligned":
        p = int(rec["point"])
        rs_abs = p - gap - ref_len
        a = max(0, rs_abs - max_shift)
        e = min(T, p + max_shift + post)
        ms = min(max_shift, rs_abs - a, e - (rs_abs + ref_len) - 1)
        X = np.asarray(traces[:, a:e], dtype=np.float32)
        ref = np.asarray(cl["mean"][rs_abs:rs_abs + ref_len], dtype=np.float32)
        al, _ = se.resync(X, ref_start=rs_abs - a, ref_len=ref_len, max_shift=ms, ref=ref)
        t = int(rec["t"]) - a - ms
        w = int(rec["w"])
        return al[:, t:t + w].sum(1).astype(np.float64)
    cand = sorted(int(cl["peaks_all"][c]) for c in sm.shuffle_group(b))
    rs_abs = cand[0] - gap - ref_len
    a = max(0, rs_abs - max_shift)
    e = min(T, cand[-1] + max_shift + 60)
    ms = min(max_shift, rs_abs - a, e - (rs_abs + ref_len) - 1)
    X = np.asarray(traces[:, a:e], dtype=np.float32)
    ref = np.asarray(cl["mean"][rs_abs:rs_abs + ref_len], dtype=np.float32)
    al, _ = se.resync(X, ref_start=rs_abs - a, ref_len=ref_len, max_shift=ms, ref=ref)
    S = np.zeros(N)
    w = int(rec["w"])
    for q in cand:
        j = q - a - ms
        lo, hi = max(0, j - w // 2), min(al.shape[1], j + w // 2 + 1)
        S += al[:, lo:hi].sum(1)
    return S


# --------------------------------------------------------------------------
# CPA sur 256 hypothèses de clé + taux de succès
# --------------------------------------------------------------------------
def _hyp_all(pts_col, model):
    x = np.asarray(pts_col, dtype=np.uint8)[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]
    return (se.HW[se.SBOX[x]] if model == "sbox" else se.HW[x]).astype(np.float64)      # [N, 256]


def _scores(H, f, idx):
    h = H[idx]
    x = f[idx]
    hm = h - h.mean(0)
    xm = x - x.mean()
    return (hm.T @ xm) / (np.sqrt((hm ** 2).sum(0) * (xm ** 2).sum()) + 1e-12)           # [256], signé


def success_curve(path, clean, bytes_=tuple(range(16)), ns=DEFAULT_NS, trials=200, max_shift=3000,
                  models=sm.MODELS, seed=0, alpha=0.9999, verbose=True, window=(1500, 24000), best=None):
    """best : résultat `best` d'un appel précédent (out["best"]) pour ne pas refaire la recherche des attaquants
    (étape la plus longue) — utile pour ajouter des métriques sans tout recalculer."""
    d = sm._load(path)
    N = len(d["traces"])
    key = np.asarray(d["key"], dtype=np.uint8)
    perm = np.random.default_rng(seed).permutation(N)              # même découpage que l'évaluateur
    h1, h2 = np.sort(perm[: N // 2]), np.sort(perm[N // 2:])
    ns = [n for n in ns if n <= len(h2)]
    if best is None:
        best = best_attackers(d, clean, bytes_, max_shift, models, seed, window)
    bytes_ = [b for b in bytes_ if b in best]

    H, F, sgn = {}, {}, {}
    for b in bytes_:
        r = best[b]
        H[b] = _hyp_all(d["pts"][:, b], r["model"])
        F[b] = feature(d, r, max_shift)
        s = _scores(H[b], F[b], h1)[key[b]]                         # signe appris sur la moitié 1
        sgn[b] = 1.0 if s >= 0 else -1.0

    if verbose:
        print("%-5s %-6s %-22s %-6s %-12s %-12s" % ("octet", "modèle", "attaquant", "rho", "N(90 %)", "N(%g)" % (100 * alpha) + " %"))
        for b in bytes_:
            r = best[b]
            lab = r.get("attacker") or ("A4(mélange,K=%d)" % r.get("K", 0))
            c = ">" if r["censored"] else " "
            print("%-5d %-6s %-22s %-6.3f %s%-11.0f %s%-11.0f" % (
                b, r["model"], lab, r["rho"], c, sm.mtd_from_rho(max(r["rho"], 1e-3), 0.90),
                c, sm.n_from_rho(max(r["rho"], 1e-3), alpha)))

    rng = np.random.default_rng(seed + 1)
    byte_ok = {n: np.zeros(len(bytes_)) for n in ns}
    full_ok = {n: 0 for n in ns}
    n_ok = {n: 0.0 for n in ns}
    bits = {n: np.zeros(trials) for n in ns}
    for n in ns:
        for tr in range(trials):
            idx = np.sort(rng.choice(h2, size=n, replace=False))
            ok = np.zeros(len(bytes_), dtype=bool)
            rk = np.zeros(len(bytes_))
            for i, b in enumerate(bytes_):
                sc = sgn[b] * _scores(H[b], F[b], idx)
                rk[i] = (sc > sc[key[b]]).sum()                     # rang de la bonne valeur (0 = trouvée)
                ok[i] = rk[i] == 0
            byte_ok[n] += ok
            full_ok[n] += bool(ok.all())
            n_ok[n] += ok.sum()
            # effort restant : taille de la « boîte » de clés à essayer = produit (rang+1) sur les octets.
            # Borne SUPÉRIEURE du rang de la clé (une énumération optimale fait mieux).
            bits[n][tr] = np.log2(1.0 + rk).sum()
    out = dict(path=path, ns=ns, trials=trials, bytes=bytes_,
               byte_sr={n: byte_ok[n] / trials for n in ns},
               full_sr={n: full_ok[n] / trials for n in ns},
               mean_bytes={n: n_ok[n] / trials for n in ns}, effort_bits=bits, best=best)
    if verbose:
        print("\n%s  (tirages dans la moitié 2 : %d traces, %d tirages par n)" % (path, len(h2), trials))
        print("%-6s %-20s %-14s %-12s %-22s %-14s" % ("n", "octets trouvés", "octet dur SR", "clé SR",
                                                   "effort restant (bits)", "effort <= 2^32"))
        for n in ns:
            e = bits[n]
            print("%-6d %5.1f / %-12d %-14.0f %-12.0f %5.1f [%4.1f - %4.1f]     %.0f %%" % (
                n, out["mean_bytes"][n], len(bytes_), 100 * out["byte_sr"][n].min(), 100 * out["full_sr"][n],
                np.median(e), np.percentile(e, 10), np.percentile(e, 90), 100 * np.mean(e <= 32)))
        for seuil, lab in ((32, "2^32 : quelques minutes sur un PC"), (48, "2^48 : gros moyens (GPU, jours)")):
            nn = next((n for n in ns if np.mean(bits[n] <= seuil) >= 0.9), None)
            print("n minimal pour que l'effort restant soit <= %s (90 %% des tirages) : %s" % (lab, nn if nn else "> %d" % ns[-1]))
        for lab, key_ in (("octet le plus FACILE", lambda n: out["byte_sr"][n].max()),
                          ("octet le plus DUR", lambda n: out["byte_sr"][n].min()),
                          ("clé complète", lambda n: out["full_sr"][n])):
            n90 = next((n for n in ns if key_(n) >= 0.9), None)
            print("n minimal pour SR >= 90 %% (%s) : %s" % (lab, n90 if n90 else "> %d" % ns[-1]))
    return out