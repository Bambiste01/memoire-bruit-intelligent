"""
monitor_collect.py — collecte pilote et analyse du moniteur SCA (firmware v6)

Utilisation dans le notebook (nt = NoiseTarget deja cree) :

    import monitor_collect as mc
    mc.run_session(nt, "normal_mixed", n_aes=300, with_scope=False, save_csv="dataset/monitor_pilot.csv")
    mc.run_session(nt, "sca_cpa_fast", n_aes=300, with_scope=True,  save_csv="dataset/monitor_pilot.csv")
    ...
    df  = mc.load_pilot("dataset/monitor_pilot.csv")
    win = mc.to_windows(df, w=32, step=16)
    mc.run_ablation(win)

Principes :
  * bruit ETEINT pendant la collecte (adaptatif off, overrides vides, intensite brute effacee) ;
  * etiquettes posees ICI, jamais deduites par le firmware ;
  * aucune valeur par defaut silencieuse : une ligne V absente leve une erreur ;
  * le temps hote (host_dt) n'est enregistre qu'a titre de controle, jamais comme feature.
"""
import os
import re
import struct
import time
import uuid

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes alignees sur le firmware
# --------------------------------------------------------------------------
F_CPU = 7_372_800            # -DF_CPU de la compilation CWLITEARM
TICK_CYCLES = 1024           # MON_TICK_SHIFT = 10
TICK_S = TICK_CYCLES / F_CPU
V_LEN = 19                   # MONITOR_RECORD_LEN

RE_V = re.compile(r"^V([0-9A-Fa-f]{%d})\s*$" % (2 * V_LEN))
RE_R = re.compile(r"^r([0-9A-Fa-f]{32})\s*$")
RE_Z = re.compile(r"^z[0-9A-Fa-f]{2}\s*\n", re.M)

FEATURES = ["n_since_key", "dt_last", "dt_ema", "dt_dev",
            "hd_last", "hd_ema", "pt_repeat", "key_changes"]


# --------------------------------------------------------------------------
# Decodage de la ligne V
# --------------------------------------------------------------------------
def decode_V(hexstr):
    b = bytes.fromhex(hexstr)
    if len(b) != V_LEN:
        raise ValueError("enregistrement V de %d octets, attendu %d" % (len(b), V_LEN))
    n, = struct.unpack(">H", b[0:2])
    dt_last, dt_ema, dt_dev = struct.unpack(">III", b[2:14])
    hd_ema_q4, = struct.unpack(">H", b[15:17])
    return dict(n_since_key=n, dt_last=dt_last, dt_ema=dt_ema, dt_dev=dt_dev,
                hd_last=b[14], hd_ema=hd_ema_q4 / 16.0,
                pt_repeat=b[17], key_changes=b[18])


def parse_V(text):
    for line in text.replace("\r", "\n").split("\n"):
        m = RE_V.match(line.strip())
        if m:
            return decode_V(m.group(1))
    raise ValueError("ligne V absente (firmware v6 flashe ?) : ...%r" % text[-120:])


def parse_ct(text):
    for line in text.replace("\r", "\n").split("\n"):
        m = RE_R.match(line.strip())
        if m:
            return bytes.fromhex(m.group(1))
    raise ValueError("ligne r absente : ...%r" % text[-120:])


# --------------------------------------------------------------------------
# Echanges serie
# --------------------------------------------------------------------------
def _decode(chunk):
    return chunk.decode("ascii", errors="replace") if isinstance(chunk, bytes) else chunk


def _flush(target, timeout_ms=50):
    try:
        target.read(timeout=timeout_ms)
    except Exception:
        pass


def _cmd(target, cmd, data, timeout=1000):
    """Commande simple avec ACK (jamais silencieuse)."""
    target.simpleserial_write(cmd, bytearray(data))
    if target.simpleserial_wait_ack(timeout=timeout) is None:
        raise RuntimeError("Pas d'ACK pour la commande %r" % cmd)


def prepare_target(target):
    """Etat de collecte : bruit eteint, mode adaptatif off, logging off, moniteur remis a zero."""
    _flush(target, 100)
    _cmd(target, "l", [0])                 # pas de ligne S
    _cmd(target, "a", [0])                 # adaptatif off
    _cmd(target, "d", [0xFF])              # pas d'override de zone
    _cmd(target, "I", [0xFF, 0, 0])        # intensites brutes effacees
    _cmd(target, "R", [0])                 # moniteur a zero


def send_pt(target, pt, scope=None, timeout_s=2.0):
    """Envoie un plaintext, lit r + V + z. Retourne (ct, features, texte brut)."""
    target.simpleserial_write("p", pt)
    if scope is not None:
        if scope.capture():
            raise RuntimeError("Timeout de capture oscilloscope")
        scope.get_last_trace()
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout_s:
        try:
            chunk = target.read(timeout=50)
        except Exception:
            chunk = None
        if chunk:
            buf += _decode(chunk).replace("\r", "\n")
        if RE_Z.search(buf):
            break
    else:
        raise RuntimeError("Reponse incomplete (pas de ligne z) : ...%r" % buf[-120:])
    return parse_ct(buf), parse_V(buf), buf


# --------------------------------------------------------------------------
# Politiques de plaintext et de cadence
# --------------------------------------------------------------------------
def pt_random(rng, st, i):
    return bytearray(rng.integers(0, 256, 16, dtype=np.uint8))


def pt_counter(rng, st, i):                         # trafic structure : compteur 128 bits
    st["c"] = (st.get("c", int(rng.integers(0, 2 ** 62))) + 1) % (1 << 128)
    return bytearray(st["c"].to_bytes(16, "big"))


def pt_ascii(rng, st, i):                           # texte imprimable
    return bytearray(rng.integers(0x20, 0x7F, 16, dtype=np.uint8))


def pt_header_mix(rng, st, i):                      # entete fixe un bloc sur deux, charge utile aleatoire
    if "hdr" not in st:
        st["hdr"] = pt_random(rng, st, 0)
    return bytearray(st["hdr"]) if rng.random() < 0.5 else pt_random(rng, st, i)


def pt_fixed_vs_random(rng, st, i):                 # test TVLA fixe contre aleatoire
    if "fix" not in st:
        st["fix"] = pt_random(rng, st, 0)
    return bytearray(st["fix"]) if rng.random() < 0.5 else pt_random(rng, st, i)


def pt_normal_mix(rng, st, i):                      # usage legitime : trois styles melanges
    pick = rng.integers(0, 3)
    return (pt_counter, pt_ascii, pt_header_mix)[pick](rng, st, i)


def gap_normal(rng):                                # rafales + pauses, jusqu'a 15 s
    u = rng.random()
    if u < 0.70:
        return rng.uniform(0.05, 0.4)
    if u < 0.95:
        return rng.uniform(0.5, 3.0)
    return rng.uniform(3.0, 15.0)


def gap_fast(rng):
    return 0.0


def gap_slow(rng):                                  # attaquant lent, recouvre l'usage normal
    return rng.uniform(0.3, 1.5)


# label = classe pour le detecteur ; subtype = metadonnee pour l'analyse
SESSION_TYPES = {
    # --- usage legitime ---
    "normal_mixed":  dict(label="normal", gap=gap_normal, pt=pt_normal_mix, key_every=(50, 200)),
    "normal_random": dict(label="normal", gap=gap_normal, pt=pt_random,     key_every=(50, 200)),
    "normal_burst":  dict(label="normal", gap=gap_fast,   pt=pt_counter,    key_every=None),
    # --- attaques par observation ---
    "sca_cpa_fast":  dict(label="sca", gap=gap_fast,   pt=pt_random,          key_every=None),
    "sca_cpa_slow":  dict(label="sca", gap=gap_slow,   pt=pt_random,          key_every=None),
    "sca_cpa_mimic": dict(label="sca", gap=gap_normal, pt=pt_random,          key_every=None),
    "sca_tvla":      dict(label="sca", gap=gap_fast,   pt=pt_fixed_vs_random, key_every=None),
    "sca_profiling": dict(label="sca", gap=gap_fast,   pt=pt_random,          key_every=(1, 10)),
}


def _next_key_change(rng, spec):
    ke = spec["key_every"]
    return None if ke is None else int(rng.integers(ke[0], ke[1] + 1))


# --------------------------------------------------------------------------
# Une session
# --------------------------------------------------------------------------
def run_session(nt, stype, n_aes=300, seed=None, with_scope=False, gap_scale=1.0,
                save_csv=None, session_id=None, verbose=True):
    """with_scope=True arme l'oscilloscope a chaque requete (cadence d'un vrai attaquant).
    Conseil : alterner True/False dans les DEUX classes pour ne pas en faire un indice."""
    spec = SESSION_TYPES[stype]
    rng = np.random.default_rng(seed)
    sid = session_id or "%s-%s" % (stype, uuid.uuid4().hex[:6])
    target = nt.target

    prepare_target(target)
    key = bytearray(rng.integers(0, 256, 16, dtype=np.uint8))
    _cmd(target, "k", key)
    since_key, expected_n = 0, 0
    next_change = _next_key_change(rng, spec)
    state, rows, last_t = {}, [], None
    mismatches = 0

    for i in range(n_aes):
        g = spec["gap"](rng) * gap_scale
        if g > 0:
            time.sleep(g)
        if next_change is not None and since_key >= next_change:
            key = bytearray(rng.integers(0, 256, 16, dtype=np.uint8))
            _cmd(target, "k", key)
            since_key, expected_n = 0, 0
            next_change = _next_key_change(rng, spec)

        pt = spec["pt"](rng, state, i)
        if with_scope:
            nt.scope.arm()
        t_send = time.time()
        _ct, f, _raw = send_pt(target, pt, scope=nt.scope if with_scope else None)

        since_key += 1
        expected_n += 1
        if f["n_since_key"] != min(expected_n, 0xFFFF):
            mismatches += 1                       # requete perdue ou compteur desynchronise

        row = dict(session_id=sid, label=spec["label"], subtype=stype, i=i,
                   with_scope=int(with_scope), gap_scale=gap_scale,
                   host_dt=(t_send - last_t) if last_t is not None else np.nan,
                   valid_dt=int(i > 0))
        row.update(f)
        rows.append(row)
        last_t = t_send

    df = pd.DataFrame(rows)
    if save_csv:
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df.to_csv(save_csv, mode="a", header=not os.path.exists(save_csv), index=False)

    if verbose:
        ok = df[(df.valid_dt == 1) & (df.dt_last > 0) & (df.host_dt > 0)]
        ratio = np.median(ok.dt_last * TICK_S / ok.host_dt) if len(ok) else float("nan")
        print("[%s] %d AES | label=%s | desync n_since_key: %d | dt firmware/hote (mediane) = %.2f"
              % (sid, len(df), spec["label"], mismatches, ratio))
        if mismatches:
            print("   ATTENTION : %d compteurs incoherents -> requetes perdues ?" % mismatches)
    return df


# --------------------------------------------------------------------------
# Analyse du pilote
# --------------------------------------------------------------------------
def load_pilot(path):
    df = pd.read_csv(path)
    return df


def to_windows(df, w=32, step=16):
    """Une fenetre = unite de classification. La 1re ligne de chaque session (dt non mesure) est ecartee."""
    out = []
    d = df[df.valid_dt == 1]
    for sid, g in d.groupby("session_id", sort=False):
        g = g.reset_index(drop=True)
        for s in range(0, len(g) - w + 1, step):
            x = g.iloc[s:s + w]
            ld = np.log1p(x.dt_last.astype(float))
            out.append(dict(
                session_id=sid, label=x.label.iloc[0], subtype=x.subtype.iloc[0],
                log_dt_mean=ld.mean(), log_dt_std=ld.std(ddof=0),
                log_dev_mean=np.log1p(x.dt_dev.astype(float)).mean(),
                hd_mean=x.hd_last.mean(), hd_std=x.hd_last.std(ddof=0),
                rep_frac=x.pt_repeat.mean(),
                log_n_since_key=np.log2(1 + x.n_since_key.iloc[-1]),
                keychg_in_win=int(x.key_changes.iloc[-1] - x.key_changes.iloc[0]),
            ))
    return pd.DataFrame(out)


FEATURE_SETS = {
    "cadence":    ["log_dt_mean", "log_dt_std", "log_dev_mean"],
    "plaintext":  ["hd_mean", "hd_std", "rep_frac"],
    "exposition": ["log_n_since_key", "keychg_in_win"],
    "tout":       ["log_dt_mean", "log_dt_std", "log_dev_mean", "hd_mean", "hd_std",
                   "rep_frac", "log_n_since_key", "keychg_in_win"],
}


def loso_eval(win, feats, group_col="session_id", depth=3):
    """Validation croisee en laissant un groupe de cote (une session, ou un sous-type d'attaque entier)."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.tree import DecisionTreeClassifier

    X = win[feats].fillna(0).values
    y = (win.label == "sca").astype(int).values
    groups = win[group_col].values
    if len(np.unique(groups)) < 2:
        raise ValueError("Il faut au moins 2 groupes (%s)" % group_col)
    pred = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    res = win[["subtype", "label"]].copy()
    res["ok"] = (pred == y)
    per_sub = res.groupby("subtype").ok.mean().round(3)
    return balanced_accuracy_score(y, pred), per_sub


def run_ablation(win, depth=3):
    """Quelles familles de features separent reellement les classes ? Deux niveaux d'exigence."""
    for gc, name in (("session_id", "session inconnue (meme sous-type connu)"),
                     ("subtype", "sous-type inconnu (style d'attaque jamais vu)")):
        print("\n=== Validation : %s ===" % name)
        for fs, feats in FEATURE_SETS.items():
            try:
                ba, per = loso_eval(win, feats, gc, depth)
            except ValueError as e:
                print("  %-11s -> %s" % (fs, e))
                continue
            print("  %-11s balanced acc = %.3f" % (fs, ba))
            if fs == "tout":
                print("     reussite par sous-type :", dict(per))
# colle ici TOUT le contenu de monitor_collect.py
# (docstring + imports + fonctions + SESSION_TYPES + run_session + ablation)
