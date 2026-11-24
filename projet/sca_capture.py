"""
sca_capture.py — version fusionnée complète
Capture + baseline_report + TVLA + évaluation A1/A2
Compatible avec NoiseTarget v6 et les anciennes campagnes.
"""
import json
import os
import re
import time
import numpy as np

import monitor_collect as mc
import sca_eval as se

try:
    from Crypto.Cipher import AES as _AES
except Exception:
    _AES = None

# --------------------------------------------------------------------------
# Utilitaires de base
# --------------------------------------------------------------------------
def to_raw(trace):
    """Trace flottante ChipWhisperer → int16 brut ADC [0, 1023]."""
    return np.clip(np.round((np.asarray(trace, dtype=np.float64) + 0.5) * 1024.0), 0, 1023).astype(np.int16)

def make_pts(n, seed):
    return np.random.default_rng(seed).integers(0, 256, (n, 16), dtype=np.uint8)

def scope_info(nt):
    info = {}
    sc_ = nt.scope
    for name, getter in (
        ("samples", lambda: sc_.adc.samples),
        ("offset", lambda: sc_.adc.offset),
        ("decimate", lambda: sc_.adc.decimate),
        ("presamples", lambda: sc_.adc.presamples),
        ("adc_src", lambda: sc_.clock.adc_src),
        ("adc_freq", lambda: sc_.clock.adc_freq),
        ("clkgen_freq", lambda: sc_.clock.clkgen_freq),
        ("gain", lambda: sc_.gain.gain),
        ("trigger", lambda: sc_.trigger.triggers),
    ):
        try:
            v = getter()
            info[name] = v if isinstance(v, (int, float, str, bool)) else str(v)
        except Exception:
            info[name] = None
    return info

def _read_until_z(target, timeout_s=2.0):
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout_s:
        try:
            chunk = target.read(timeout=50)
        except Exception:
            chunk = None
        if chunk:
            buf += mc._decode(chunk).replace("\r", "\n")
        if mc.RE_Z.search(buf):
            return buf
    raise RuntimeError("pas de ligne z : ...%r" % buf[-80:])

RE_P = re.compile(r"P([0-9A-Fa-f]{32})\n")
RE_T = re.compile(r"T([0-9A-Fa-f]{56})\n")


def parse_plog(buf):
    """Ligne 'P' du firmware v7 (journal de décision, 16 octets) -> np.uint8[16] ou None."""
    m = RE_P.search(buf)
    return np.frombuffer(bytes.fromhex(m.group(1)), dtype=np.uint8) if m else None


def decode_plog(plog):
    """plog [N, 16] -> dict de colonnes (level, x[N,6], decide_cycles, flags, exhausted, session_class)."""
    p = np.asarray(plog, dtype=np.uint8).reshape(-1, 16).astype(np.int64)
    x = np.stack([(p[:, 1 + 2 * i] << 8) | p[:, 2 + 2 * i] for i in range(6)], axis=1)
    return dict(level=p[:, 0], x=x, decide_cycles=(p[:, 13] << 8) | p[:, 14], flags=p[:, 15],
                exhausted=(p[:, 15] & 1) == 1, session_class=(p[:, 15] >> 4) & 0x7)


def capture_one(nt, pt, retries=3, with_trig=False, with_plog=False):
    """with_plog=True : renvoie en plus le journal 'P' (None hors mode adaptatif)."""
    last = None
    for _ in range(retries):
        try:
            nt.scope.arm()
            nt.target.simpleserial_write("p", bytearray(pt))
            if nt.scope.capture():
                raise RuntimeError("timeout de capture oscilloscope")
            tr = nt.scope.get_last_trace()
            try:
                trig = int(nt.scope.adc.trig_count)
            except Exception:
                trig = 0
            buf = _read_until_z(nt.target)
            ct = np.frombuffer(mc.parse_ct(buf), dtype=np.uint8)
            out = (to_raw(tr), ct, trig) if with_trig else (to_raw(tr), ct)
            return out + (parse_plog(buf),) if with_plog else out
        except Exception as e:
            last = e
            mc._flush(nt.target, 100)
    raise RuntimeError("echec de capture apres %d essais : %s" % (retries, last))

def expected_ct(key, pts):
    if _AES is None:
        return None
    c = _AES.new(bytes(key), _AES.MODE_ECB)
    return np.frombuffer(c.encrypt(np.ascontiguousarray(pts).tobytes()), dtype=np.uint8).reshape(-1, 16)

# --------------------------------------------------------------------------
# Capture de campagne
# --------------------------------------------------------------------------
def capture_campaign(nt, name, n_traces, key, pts=None, seed=0, outdir="traces", chunk=1000,
                     setup=None, meta=None, verbose=True):
    path = os.path.join(outdir, name)
    os.makedirs(path, exist_ok=True)
    key = bytes(key)
    pts = make_pts(n_traces, seed) if pts is None else np.ascontiguousarray(pts, dtype=np.uint8)
    if len(pts) != n_traces:
        raise ValueError("len(pts) != n_traces")
    n_chunks = (n_traces + chunk - 1) // chunk
    cpath = lambda i: os.path.join(path, "chunk_%05d.npz" % i)
    todo = [i for i in range(n_chunks) if not os.path.exists(cpath(i))]

    if os.path.exists(os.path.join(path, "meta.json")):     # capture terminée (ses blocs sont effacés à la fin)
        if verbose:
            print("[%s] deja complete -> rien a faire" % name)
        return path

    if todo:
        mc.prepare_target(nt.target)
        mc._cmd(nt.target, "k", bytearray(key))
        if setup is not None:
            setup(nt)
        if verbose and len(todo) < n_chunks:
            print("[%s] reprise : %d/%d blocs deja presents" % (name, n_chunks - len(todo), n_chunks))

    t_start, done = time.time(), 0
    for i in todo:
        a, b = i * chunk, min(n_traces, (i + 1) * chunk)
        trs, cts, trg, plg = [], [], [], []
        for j in range(a, b):
            tr, ct, tg, pl = capture_one(nt, pts[j], with_trig=True, with_plog=True)
            trs.append(tr); cts.append(ct); trg.append(tg)
            plg.append(pl if pl is not None else np.full(16, 0xFF, dtype=np.uint8))
        trs, cts, trg = np.stack(trs), np.stack(cts), np.array(trg, dtype=np.uint32)
        plg = np.stack(plg)
        exp = expected_ct(key, pts[a:b])
        bad = int((cts != exp).any(axis=1).sum()) if exp is not None else -1
        np.savez(cpath(i), traces=trs, cts=cts, trig=trg, plog=plg, ct_bad=bad)
        done += b - a
        if verbose:
            el = time.time() - t_start
            eta = el / done * (len(todo) * chunk - done) if done else 0
            print("[%s] bloc %d/%d | %d traces | chiffres faux: %s | %.0f ms/trace | reste ~%.0f min"
                  % (name, i + 1, n_chunks, b - a, bad if bad >= 0 else "n/a", el / done * 1000, eta / 60))
        if bad > 0:
            print("   ATTENTION : %d chiffres incorrects dans ce bloc" % bad)

    first = np.load(cpath(0))
    T = first["traces"].shape[1]
    mm = np.lib.format.open_memmap(os.path.join(path, "traces.npy"), mode="w+", dtype=np.int16, shape=(n_traces, T))
    cts_all = np.empty((n_traces, 16), dtype=np.uint8)
    trig_all = np.zeros(n_traces, dtype=np.uint32)
    plog_all = np.full((n_traces, 16), 0xFF, dtype=np.uint8)
    bad_total = 0
    for i in range(n_chunks):
        z = np.load(cpath(i))
        a = i * chunk
        mm[a:a + len(z["traces"])] = z["traces"]
        cts_all[a:a + len(z["cts"])] = z["cts"]
        trig_all[a:a + len(z["trig"])] = z["trig"]
        if "plog" in z.files:
            plog_all[a:a + len(z["plog"])] = z["plog"]
        bad_total += int(z["ct_bad"]) if int(z["ct_bad"]) > 0 else 0
    mm.flush(); del mm
    np.save(os.path.join(path, "pts.npy"), pts)
    np.save(os.path.join(path, "cts.npy"), cts_all)
    np.save(os.path.join(path, "trig.npy"), trig_all)
    if (plog_all[:, 0] != 0xFF).any():                     # mode adaptatif : journal de décision
        np.save(os.path.join(path, "plog.npy"), plog_all)
    info = dict(name=name, n_traces=int(n_traces), n_samples=int(T), key=key.hex(), seed=int(seed),
                ct_errors=int(bad_total), trig_mean=float(trig_all.mean()),
                created=time.strftime("%Y-%m-%d %H:%M:%S"),
                raw_to_float="trace = raw/1024 - 0.5", scope=scope_info(nt), **(meta or {}))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    for i in range(n_chunks):
        os.remove(cpath(i))
    if verbose:
        print("[%s] OK : %d traces x %d echantillons | chiffres faux: %d | %s"
              % (name, n_traces, T, bad_total, path))
    return path

def load_campaign(path, mmap=True):
    d = dict(
        traces=np.load(os.path.join(path, "traces.npy"), mmap_mode="r" if mmap else None),
        pts=np.load(os.path.join(path, "pts.npy")),
        cts=np.load(os.path.join(path, "cts.npy"))
    )
    with open(os.path.join(path, "meta.json")) as f:
        d["meta"] = json.load(f)
    d["key"] = np.frombuffer(bytes.fromhex(d["meta"]["key"]), dtype=np.uint8)
    tp = os.path.join(path, "trig.npy")
    d["trig"] = np.load(tp) if os.path.exists(tp) else None
    pp = os.path.join(path, "plog.npy")
    d["plog"] = np.load(pp) if os.path.exists(pp) else None
    return d

def save_existing(path, traces, pts, key, meta=None):
    """Étape 0 : sauvegarde de traces déjà en mémoire."""
    os.makedirs(path, exist_ok=True)
    tr = np.asarray(traces)
    if tr.dtype != np.int16:
        tr = to_raw(tr)
    np.save(os.path.join(path, "traces.npy"), tr)
    np.save(os.path.join(path, "pts.npy"), np.ascontiguousarray(pts, dtype=np.uint8))
    np.save(os.path.join(path, "cts.npy"), np.zeros((len(tr), 16), dtype=np.uint8))
    info = dict(name=os.path.basename(path), n_traces=int(len(tr)), n_samples=int(tr.shape[1]),
                key=bytes(key).hex(), imported=True, **(meta or {}))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(info, f, indent=2)
    return path

# --------------------------------------------------------------------------
# Rapport de baseline
# --------------------------------------------------------------------------
def _peaks(rho, N, thr_sigma=5.0):
    thr = thr_sigma / np.sqrt(N)
    rows = []
    for b in range(rho.shape[0]):
        a = np.abs(rho[b])
        t = int(np.argmax(a))
        floor = 1.4826 * np.median(a)
        rows.append(dict(byte=b, sample=t, rho=float(rho[b, t]), abs_rho=float(a[t]),
                         floor=float(floor), ratio=float(a[t] / (floor + 1e-12)),
                         detectable=bool(a[t] > thr)))
    return rows, thr

def baseline_report(path, crop=300, n_perm=20, n_max_mtd=None, plot=True, edge_margin=200):
    d = load_campaign(path)
    tr, pts, key = d["traces"], d["pts"], d["key"]
    N, T = tr.shape
    rep = dict(path=path, n_traces=int(N), n_samples=int(T), ct_errors=int(d["meta"].get("ct_errors", -1)))

    sub = np.asarray(tr[::max(1, N // 500)], dtype=np.int16)
    rep["adc_min"], rep["adc_max"] = int(sub.min()), int(sub.max())
    rep["adc_saturated_frac"] = float(((sub <= 0) | (sub >= 1023)).mean())

    rho = se.corr_all_bytes(tr, pts, key)
    peaks, thr = _peaks(rho, N)
    for p in peaks:
        p["in_window"] = bool(edge_margin <= p["sample"] < T - edge_margin)
    rep["peaks"], rep["rho_threshold"] = peaks, float(thr)

    # Stabilité
    h = N // 2
    rho1 = se.corr_all_bytes(tr[:h], pts[:h], key)
    rho2 = se.corr_all_bytes(tr[h:], pts[h:], key)
    p1, _ = _peaks(rho1, h)
    p2, _ = _peaks(rho2, N - h)
    stab = []
    for a_, b_ in zip(p1, p2):
        if a_["detectable"] and b_["detectable"]:
            stab.append(dict(byte=a_["byte"],
                             d_sample=abs(a_["sample"] - b_["sample"]),
                             rho_ratio=min(a_["abs_rho"], b_["abs_rho"]) / max(a_["abs_rho"], b_["abs_rho"])))
    rep["stability"] = stab

    # MTD
    mtd = {}
    for p in peaks:
        if not (p["detectable"] and p["in_window"]):
            continue
        a_ = max(0, p["sample"] - crop)
        b_ = min(T, p["sample"] + crop)
        X = np.asarray(tr[:, a_:b_], dtype=np.float32)
        r = se.ge_sr_mtd(X, pts, key, byte=p["byte"], n_perm=n_perm,
                         checkpoints=se.default_checkpoints(n_max_mtd or N // 2))
        mtd[p["byte"]] = dict(mtd_median=r["mtd_median"],
                              ge_last=float(r["ge"][-1]),
                              protection=se.protection_score(r),
                              crop=[int(a_), int(b_)])
    rep["mtd"] = mtd

    # Affichage
    print("=== Baseline %s : %d traces x %d echantillons ===" % (os.path.basename(path), N, T))
    print("chiffres faux : %s | ADC [%d, %d] | saturation %.4f%% | seuil rho = %.4f"
          % (rep["ct_errors"], rep["adc_min"], rep["adc_max"], 100 * rep["adc_saturated_frac"], thr))
    print("%-5s %-8s %-9s %-7s %-9s %-10s" % ("octet", "echant.", "|rho|", "ratio", "detect.", "dans fenetre"))
    for p in peaks:
        print("%-5d %-8d %-9.4f %-7.1f %-9s %-10s" % (
            p["byte"], p["sample"], p["abs_rho"], p["ratio"],
            "oui" if p["detectable"] else "non",
            "oui" if p["in_window"] else "BORD"))
    if stab:
        ok = [s for s in stab if s["d_sample"] <= 2 and s["rho_ratio"] > 0.7]
        print("stabilite 1re/2e moitie : %d/%d octets stables" % (len(ok), len(stab)))
    if mtd:
        print("MTD de reference :")
        for b, m in mtd.items():
            print("  octet %2d : MTD=%s | GE@N/2=%.1f | protection=%.3f"
                  % (b, ("%d" % m["mtd_median"]) if np.isfinite(m["mtd_median"]) else "inf",
                     m["ge_last"], m["protection"]))

    with open(os.path.join(path, "baseline_report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    return rep

# --------------------------------------------------------------------------
# TVLA
# --------------------------------------------------------------------------
def capture_tvla(nt, name, n_per_class=5000, key=None, fixed_pt=None, seed=7,
                 outdir="traces", chunk=1000, setup=None, verbose=True):
    rng = np.random.default_rng(seed)
    fixed = rng.integers(0, 256, 16, dtype=np.uint8) if fixed_pt is None else np.asarray(fixed_pt, dtype=np.uint8)
    n = 2 * n_per_class
    labels = np.zeros(n, dtype=np.uint8)
    labels[:n_per_class] = 1
    rng.shuffle(labels)
    pts = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    pts[labels == 1] = fixed
    path = capture_campaign(nt, name, n, key, pts=pts, seed=seed, outdir=outdir,
                            chunk=chunk, setup=setup,
                            meta=dict(kind="tvla", fixed_pt=fixed.tobytes().hex()), verbose=verbose)
    np.save(os.path.join(path, "labels.npy"), labels)
    return path

def tvla_report(path, threshold=4.5, plot=True):
    d = load_campaign(path)
    labels = np.load(os.path.join(path, "labels.npy"))
    tr = d["traces"]
    A = np.zeros(tr.shape[1]); B = np.zeros(tr.shape[1])
    A2 = np.zeros_like(A); B2 = np.zeros_like(B)
    nA = nB = 0
    for s in range(0, len(tr), 1000):
        X = np.asarray(tr[s:s+1000], dtype=np.float64)
        l = labels[s:s+1000]
        Xa, Xb = X[l == 1], X[l == 0]
        A += Xa.sum(0); A2 += (Xa**2).sum(0); nA += len(Xa)
        B += Xb.sum(0); B2 += (Xb**2).sum(0); nB += len(Xb)
    ma, mb = A/nA, B/nB
    va = (A2/nA - ma**2) * nA/(nA-1)
    vb = (B2/nB - mb**2) * nB/(nB-1)
    t = (ma - mb) / np.sqrt(va/nA + vb/nB + 1e-18)
    leaking = np.flatnonzero(np.abs(t) > threshold)
    print("TVLA %s : max|t| = %.1f @ %d | %d echantillons > %.1f (%.2f%%)"
          % (os.path.basename(path), np.abs(t).max(), int(np.argmax(np.abs(t))),
             len(leaking), threshold, 100*len(leaking)/len(t)))
    np.save(os.path.join(path, "tvla_t.npy"), t)
    return t

# --------------------------------------------------------------------------
# Setup & évaluation A1/A2
# --------------------------------------------------------------------------
def make_setup(forces=(), raw=None, round_mask=0x07FF, adaptive=False, leak_limit_q16=None, clear=True):
    """Configure la cible avant une campagne (firmware v7).
    forces : [(zone, générateur, intensité)] ; raw : {générateur: cycles} ; round_mask : bit r = ronde r.
    clear=True remet d'abord la cible à zéro (aucun forçage, aucune intensité brute, toutes rondes) :
    sans cela, la config d'une campagne précédente persiste (source d'erreurs dans les notebooks v5/v6)."""
    def _setup(nt):
        if clear:
            mc._cmd(nt.target, "a", bytearray([0]))
            mc._cmd(nt.target, "d", bytearray([0xFF]))
            mc._cmd(nt.target, "I", bytearray([0xFF, 0, 0]))
        for z, g, i in forces:
            mc._cmd(nt.target, "c", bytearray([int(z), int(g), int(i)]))
        for g, cy in (raw or {}).items():
            mc._cmd(nt.target, "I", bytearray([int(g), (int(cy)>>8)&0xFF, int(cy)&0xFF]))
        m = int(round_mask) & 0x07FF
        mc._cmd(nt.target, "M", bytearray([m >> 8, m & 0xFF]))
        if leak_limit_q16 is not None:
            v = int(leak_limit_q16)
            mc._cmd(nt.target, "B", bytearray([(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]))
        if adaptive:
            mc._cmd(nt.target, "a", bytearray([1]))
    return _setup


def read_zone_timestamps(nt, key, pt=None, rounds=range(11)):
    """Horodatage (cycles depuis le début du chiffrement) de chaque (ronde, zone) sur UN chiffrement.
    À faire SANS bruit (aucun forçage) : sinon les horodatages incluent les délais aléatoires.
    Retourne {(ronde, zone): cycles ou None}."""
    pt = np.zeros(16, dtype=np.uint8) if pt is None else np.asarray(pt, dtype=np.uint8)
    mc._cmd(nt.target, "k", bytearray(bytes(key)))
    mc._cmd(nt.target, "T", bytearray([0xFF]))
    nt.target.simpleserial_write("p", bytearray(pt))
    _read_until_z(nt.target)
    out = {}
    for r in rounds:
        nt.target.simpleserial_write("T", bytearray([r]))
        buf = _read_until_z(nt.target)
        m = RE_T.search(buf)
        if not m:
            raise RuntimeError("pas de ligne T pour la ronde %d : %r" % (r, buf[-80:]))
        raw = bytes.fromhex(m.group(1))
        for z in range(7):
            v = int.from_bytes(raw[4 * z:4 * z + 4], "big")
            out[(r, z)] = None if v == 0xFFFFFFFF else v
    return out

def evaluate_campaign(path, bytes_=(0,5,10,15), window=(3000,11000),
                      realign=dict(ref_start=3500, ref_len=800, max_shift=500),
                      n_perm=20, use_n=None):
    d = load_campaign(path)
    tr, pts, key = d["traces"], d["pts"], d["key"]
    N = len(tr) if use_n is None else min(len(tr), use_n)
    a0, b0 = window
    X = np.asarray(tr[:N, a0:b0], dtype=np.float32)
    out = dict(path=path, n=int(N),
               cycles_mean=float(d["trig"][:N].mean()) if d["trig"] is not None else float("nan"))

    def attack(Xm, tag):
        rho = se.corr_all_bytes(Xm, pts[:N], key)
        per = {}
        for b in bytes_:
            r = se.ge_sr_mtd(Xm, pts[:N], key, byte=b, n_perm=n_perm)
            per[b] = dict(mtd=r["mtd_median"], ge=float(r["ge"][-1]),
                          prot=se.protection_score(r), rho=float(np.abs(rho[b]).max()))
        out[tag] = dict(per_byte=per,
                        mtd_min=float(min(v["mtd"] for v in per.values())),
                        prot_mean=float(np.mean([v["prot"] for v in per.values()])),
                        rho_mean=float(np.mean([v["rho"] for v in per.values()])))

    attack(X, "A1")
    out["A2"] = None
    if realign is not None:
        rl = dict(realign)
        if rl["ref_start"] - rl["max_shift"] < a0:
            raise ValueError("ref_start - max_shift < debut de fenetre")
        rl["ref_start"] -= a0
        al, lags = se.resync(X, **rl)
        out["lags"] = dict(min=int(lags.min()), max=int(lags.max()), std=float(lags.std()))
        attack(al, "A2")
    return out

def compare_campaigns(paths, baseline=None, **kw):
    rows = [evaluate_campaign(p, **kw) for p in paths]
    ref = evaluate_campaign(baseline, **kw)["cycles_mean"] if baseline else None
    hdr = "%-22s %9s %8s | %-8s %-7s %-6s | %-8s %-7s %-6s | %s" % (
        "config", "cycles", "surcout", "A1 MTD", "A1 prot", "A1 rho",
        "A2 MTD", "A2 prot", "A2 rho", "lags")
    print(hdr); print("-" * len(hdr))
    f = lambda v: ("inf" if not np.isfinite(v) else "%d" % v)
    for p, r in zip(paths, rows):
        ov = ("%+.1f%%" % (100*(r["cycles_mean"]/ref - 1))) if ref and np.isfinite(r["cycles_mean"]) else "n/a"
        a1, a2 = r["A1"], r["A2"]
        lg = r.get("lags")
        print("%-22s %9.0f %8s | %-8s %-7.3f %-6.3f | %-8s %-7s %-6s | %s" % (
            os.path.basename(p), r["cycles_mean"], ov,
            f(a1["mtd_min"]), a1["prot_mean"], a1["rho_mean"],
            f(a2["mtd_min"]) if a2 else "-",
            ("%.3f" % a2["prot_mean"]) if a2 else "-",
            ("%.3f" % a2["rho_mean"]) if a2 else "-",
            ("%d/%d/%.1f" % (lg["min"], lg["max"], lg["std"])) if lg else "-"))
    return rows
