"""
sca_eval.py — évaluation de la sécurité SCA pour l'optimisation du générateur de bruit

Entrées (format indépendant du banc) :
    traces : ndarray [N, T]   (une ligne par chiffrement, fenêtre recadrée autour de la fuite)
    pts    : ndarray [N, 16]  uint8, plaintexts
    key    : sequence de 16 octets (clé réelle, connue de l'évaluateur)

Pourquoi ce module (leçons de la campagne précédente) :
  * le rang à N fixe a un écart-type ~60 : trop bruité pour piloter une optimisation ;
  * le MTD vaut `inf` pour presque toute config protégée : aucune information de gradient ;
  * l'efficacité d'un jitter dépend de la capacité de réalignement de l'attaquant.
Donc : GE/SR par ré-échantillonnage (bootstrap), score CONTINU `protection` dans [0, 1],
et deux attaquants (statique A1, réaligné A2).

Références : Standaert, Malkin, Yung (EUROCRYPT 2009) pour GE/SR ; Brier, Clavier, Olivier (CHES 2004)
pour la CPA ; Goodwill et al. (2011) pour le test de fuite fixe-contre-aléatoire.
"""
import numpy as np


# --------------------------------------------------------------------------
# S-box AES (construite, pas recopiée) et poids de Hamming
# --------------------------------------------------------------------------
def _build_sbox():
    def gmul(a, b):
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p

    inv = [0] * 256
    for x in range(1, 256):
        for y in range(1, 256):
            if gmul(x, y) == 1:
                inv[x] = y
                break
    sbox = []
    for x in range(256):
        b = inv[x]
        s = b
        for r in (1, 2, 3, 4):
            s ^= ((b << r) | (b >> (8 - r))) & 0xFF
        sbox.append(s ^ 0x63)
    return np.array(sbox, dtype=np.uint8)


SBOX = _build_sbox()
assert SBOX[0x00] == 0x63 and SBOX[0x01] == 0x7C and SBOX[0x53] == 0xED
HW = np.array([bin(i).count("1") for i in range(256)], dtype=np.float64)
_KEYS = np.arange(256, dtype=np.uint8)


def hypotheses(pts, byte):
    """H[n, k] = HW(SBOX[pt[n, byte] xor k]) — modèle de fuite classique."""
    return HW[SBOX[pts[:, byte][:, None] ^ _KEYS[None, :]]]


# --------------------------------------------------------------------------
# CPA cumulative (statistiques suffisantes : un seul passage par ordre de traces)
# --------------------------------------------------------------------------
def cpa_scores_cumulative(traces, pts, byte, order, checkpoints):
    """Pour chaque checkpoint c : score[k] = max_t |rho(H[:, k], traces[:, t])| sur les c premières traces de `order`.
    Les traces sont lues par blocs (compatible tableaux int16 et np.memmap : pas de copie complète)."""
    T = traces.shape[1]
    sh = np.zeros(256); sh2 = np.zeros(256)
    st = np.zeros(T);   st2 = np.zeros(T)
    sht = np.zeros((256, T))
    out = np.empty((len(checkpoints), 256))
    prev = 0
    for i, c in enumerate(checkpoints):
        idx = np.sort(order[prev:c])                       # tri : lecture memmap plus rapide
        Hc = hypotheses(pts[idx], byte)
        Xc = np.asarray(traces[idx], dtype=np.float64)
        sh += Hc.sum(0); sh2 += (Hc ** 2).sum(0)
        st += Xc.sum(0); st2 += (Xc ** 2).sum(0)
        sht += Hc.T @ Xc
        n = c
        cov = n * sht - sh[:, None] * st[None, :]
        vh = n * sh2 - sh ** 2
        vt = n * st2 - st ** 2
        den = np.sqrt(np.maximum(vh[:, None], 1e-12) * np.maximum(vt[None, :], 1e-12))
        out[i] = np.max(np.abs(cov / den), axis=1)
        prev = c
    return out


def corr_all_bytes(traces, pts, key, chunk=1000):
    """rho[b, t] = corrélation de HW(SBOX[pt_b ^ k_b]) (clé CORRECTE) avec chaque échantillon, pour les 16 octets,
    en un seul passage sur les traces."""
    N, T = traces.shape
    key = np.asarray(key, dtype=np.uint8)
    Hall = np.stack([HW[SBOX[pts[:, b] ^ key[b]]] for b in range(16)])      # [16, N]
    sh = Hall.sum(1); sh2 = (Hall ** 2).sum(1)
    st = np.zeros(T); st2 = np.zeros(T); sht = np.zeros((16, T))
    for s in range(0, N, chunk):
        X = np.asarray(traces[s:s + chunk], dtype=np.float64)
        st += X.sum(0); st2 += (X ** 2).sum(0)
        sht += Hall[:, s:s + chunk] @ X
    cov = N * sht - sh[:, None] * st[None, :]
    den = np.sqrt(np.maximum((N * sh2 - sh ** 2)[:, None], 1e-12) * np.maximum((N * st2 - st ** 2)[None, :], 1e-12))
    return cov / den


def default_checkpoints(n_max, n_min=5, count=20):
    """Points de contrôle géométriques. n_min=5 : sur une cible à très fort signal (rho ~ 0,9) le MTD sans bruit
    est inférieur à 50 traces ; un plancher à 50 le rendrait invisible (MTD = 'plancher')."""
    cp = np.unique(np.round(np.geomspace(n_min, n_max, count)).astype(int))
    return cp[(cp >= 2) & (cp <= n_max)]


def ge_sr_mtd(traces, pts, key, byte=0, checkpoints=None, n_perm=30, seed=0):
    """Guessing entropy, success rate et MTD par ré-échantillonnage (n_perm ordres aléatoires).

    ATTENTION : les sous-ensembles se recouvrent ; ne pas interpréter les checkpoints > N/2.
    """
    N = len(traces)
    if checkpoints is None:
        checkpoints = default_checkpoints(N // 2)
    checkpoints = np.asarray(checkpoints)
    if checkpoints.max() > N:
        raise ValueError("checkpoint > nombre de traces")
    rng = np.random.default_rng(seed)
    kb = int(key[byte])
    ranks = np.empty((n_perm, len(checkpoints)), dtype=int)
    for r in range(n_perm):
        sc = cpa_scores_cumulative(traces, pts, byte, rng.permutation(N), checkpoints)
        ranks[r] = (sc > sc[:, [kb]]).sum(1)
    ge = ranks.mean(0)
    sr = (ranks == 0).mean(0)
    mtds = []
    for r in range(n_perm):
        ok = ranks[r] == 0
        suffix = np.logical_and.accumulate(ok[::-1])[::-1]
        mtds.append(checkpoints[np.argmax(suffix)] if suffix.any() else np.inf)
    return dict(checkpoints=checkpoints, ge=ge, sr=sr, ranks=ranks,
                mtd_median=float(np.median(mtds)), mtd_all=np.array(mtds, dtype=float),
                n_reliable=N // 2)


def protection_score(res):
    """Score CONTINU dans [0,1] : 0 = clé trouvée tout de suite, 1 = aucune information (GE ~ hasard).
    Moyenne de log2(1+GE) sur les checkpoints fiables, normalisée par le hasard (GE = 127,5)."""
    cp, ge = res["checkpoints"], res["ge"]
    m = cp <= res["n_reliable"]
    return float(min(1.0, np.mean(np.log2(1.0 + ge[m])) / np.log2(1.0 + 127.5)))


# --------------------------------------------------------------------------
# Test de fuite fixe-contre-aléatoire (Welch), sans modèle de fuite
# --------------------------------------------------------------------------
def tvla(a, b):
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    return (ma - mb) / np.sqrt(va / len(a) + vb / len(b) + 1e-18)


# --------------------------------------------------------------------------
# Attaquant réaligné (A2) : corrélation croisée normalisée sur un motif indépendant des données
# --------------------------------------------------------------------------
def resync(traces, ref_start, ref_len, max_shift, iters=2, ref=None):
    """Aligne chaque trace sur un motif de référence.

    Le motif [ref_start, ref_start+ref_len) doit être INDEPENDANT DES DONNEES (début de ronde, etc.) et
    situé juste avant la fuite visée. Retourne (aligned [N, T-2*max_shift], lags [N]) avec
    aligned[n, j] = traces[n, j + max_shift + lags[n]], lags dans [-max_shift, +max_shift].
    """
    X = traces.astype(np.float32)
    N, T = X.shape
    if ref_start - max_shift < 0 or ref_start + ref_len + max_shift > T:
        raise ValueError("motif + décalage hors de la fenêtre de capture")
    lo = ref_start - max_shift
    span = ref_len + 2 * max_shift
    nl = 2 * max_shift + 1
    if ref is None:
        ref = np.median(X[: min(N, 200), ref_start:ref_start + ref_len], axis=0)
    lags = np.zeros(N, dtype=int)
    chunk = max(1, int(3e7 // (nl * ref_len)))
    for _ in range(iters):
        r = ref - ref.mean()
        r /= (np.linalg.norm(r) + 1e-12)
        for s in range(0, N, chunk):
            seg = X[s:s + chunk, lo:lo + span]
            W = np.lib.stride_tricks.sliding_window_view(seg, ref_len, axis=1)   # [c, nl, L]
            Wc = W - W.mean(axis=2, keepdims=True)
            ncc = (Wc @ r) / (np.linalg.norm(Wc, axis=2) + 1e-12)
            lags[s:s + chunk] = np.argmax(ncc, axis=1) - max_shift
        idx = (ref_start + lags)[:, None] + np.arange(ref_len)[None, :]
        ref = X[np.arange(N)[:, None], idx].mean(axis=0)
    cols = (max_shift + lags)[:, None] + np.arange(T - 2 * max_shift)[None, :]
    return X[np.arange(N)[:, None], cols], lags


# --------------------------------------------------------------------------
# Évaluation d'une configuration de bruit : A1 (statique) et A2 (réaligné)
# --------------------------------------------------------------------------
def evaluate_config(traces, pts, key, byte=0, realign=None, n_perm=30, seed=0):
    """realign = dict(ref_start=..., ref_len=..., max_shift=...) ou None.
    Retourne {'A1': res, 'A2': res|None} avec res['protection'] ajouté."""
    out = {}
    r1 = ge_sr_mtd(traces, pts, key, byte, n_perm=n_perm, seed=seed)
    r1["protection"] = protection_score(r1)
    out["A1"] = r1
    out["A2"] = None
    if realign is not None:
        al, lags = resync(traces, **realign)
        r2 = ge_sr_mtd(al, pts, key, byte, n_perm=n_perm, seed=seed)
        r2["protection"] = protection_score(r2)
        r2["lags"] = lags
        out["A2"] = r2
    return out