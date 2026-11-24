"""
acceleration.py — réalignement rapide, résultat IDENTIQUE à sca_eval.resync (vérifié : 100 % des décalages égaux).

Principe : la corrélation normalisée entre le motif et chaque fenêtre glissante est calculée
  - au numérateur par une convolution FFT (le motif étant centré, sum((w - mean(w)) * r) = sum(w * r)) ;
  - au dénominateur par des sommes cumulées de x et x² (écart-type glissant en O(T)).
L'ancienne version matérialisait un tableau [traces, décalages, 60] puis le recentrait : c'était 78 % du temps
de best_attackers (profil sur delais_avant_20k, octet 0, 2000 traces).

Utilisation (au début d'un script ou d'un notebook, APRÈS les imports) :
    import acceleration; acceleration.install()
"""
import numpy as np
from scipy.signal import fftconvolve


def resync_fast(traces, ref_start, ref_len, max_shift, iters=2, ref=None):
    X = traces.astype(np.float32)
    N, T = X.shape
    if ref_start - max_shift < 0 or ref_start + ref_len + max_shift > T:
        raise ValueError("motif + décalage hors de la fenêtre de capture")
    lo = ref_start - max_shift
    span = ref_len + 2 * max_shift
    if ref is None:
        ref = np.median(X[: min(N, 200), ref_start:ref_start + ref_len], axis=0)
    lags = np.zeros(N, dtype=int)
    # Traitement par blocs de traces : mémoire bornée (~50 Mo par bloc de 1000 traces au lieu de ~11 Go pour
    # 20 000 traces d'un coup), résultat identique. Indispensable quand plusieurs processus tournent en parallèle.
    chunk = 1000
    for _ in range(iters):
        r = (np.asarray(ref, np.float64) - np.mean(ref))
        r /= (np.linalg.norm(r) + 1e-12)
        for s in range(0, N, chunk):
            seg = X[s:s + chunk, lo:lo + span].astype(np.float64)
            z = np.zeros((len(seg), 1))
            c1 = np.concatenate([z, np.cumsum(seg, 1)], 1)
            c2 = np.concatenate([z, np.cumsum(seg ** 2, 1)], 1)
            S1 = c1[:, ref_len:] - c1[:, :-ref_len]
            S2 = c2[:, ref_len:] - c2[:, :-ref_len]
            den = np.sqrt(np.maximum(S2 - S1 ** 2 / ref_len, 0.0)) + 1e-12
            num = fftconvolve(seg, r[::-1][None, :], mode="valid", axes=1)
            lags[s:s + chunk] = np.argmax(num / den, axis=1) - max_shift
        idx = (ref_start + lags)[:, None] + np.arange(ref_len)[None, :]
        ref = X[np.arange(N)[:, None], idx].mean(axis=0)
    cols = (max_shift + lags)[:, None] + np.arange(T - 2 * max_shift)[None, :]
    return X[np.arange(N)[:, None], cols], lags


def install():
    """Remplace sca_eval.resync par la version rapide (sca_metrics et verify_n l'appellent via se.resync)."""
    import sca_eval as se
    se.resync = resync_fast
    return se
