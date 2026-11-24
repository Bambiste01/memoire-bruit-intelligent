"""
fautes_v10.py — signaux de faute du firmware v10.1 (lignes 'H' et 'G') et opérations sans oscilloscope.

Ligne 'G' (8 octets, firmware v10.1, émise avec 'H' quand le rapport de faute est actif : commande 'W' [mode, 1]) :
  net_cycles    durée du dernier chiffrement moins les cycles de délais insérés (sur la carte, compteur DWT)
  total_cycles  durée du dernier chiffrement
  round_err_recent, flags (bit0 erreur de ronde dans l'opération, bit1 désaccord de redondance, bit2 redondance)
Ligne 'H' (8 octets, v9) : detect_recent, boots_recent, drapeaux et cause du reset, detect_total, redundant_runs.

Ces signaux sont les entrées candidates du classifieur de fautes (étape 4). Aucune récupération de clé ici.
"""
import re

import numpy as np

RE_G = re.compile(r"G([0-9A-Fa-f]{16})\n")
RE_H = re.compile(r"H([0-9A-Fa-f]{16})\n")
RE_R = re.compile(r"r([0-9A-Fa-f]{32})\n")
RE_P = re.compile(r"P([0-9A-Fa-f]{32})\n")


def parse_g(buf):
    m = RE_G.search(buf)
    if not m:
        return None
    r = bytes.fromhex(m.group(1))
    return dict(net_cycles=(r[0] << 16) | (r[1] << 8) | r[2], total_cycles=(r[3] << 16) | (r[4] << 8) | r[5],
                round_err_recent=r[6], round_err_op=bool(r[7] & 1), mismatch=bool(r[7] & 2), redundant=bool(r[7] & 4))


def parse_h(buf):
    m = RE_H.search(buf)
    if not m:
        return None
    r = bytes.fromhex(m.group(1))
    return dict(detect_recent=(r[0] << 8) | r[1], boots_recent=r[2], mismatch=bool(r[3] & 1),
                redundant=bool(r[3] & 2), cold_boot=bool(r[3] & 4), reset_cause=r[3] >> 4,
                detect_total=(r[4] << 8) | r[5], redundant_runs=(r[6] << 8) | r[7])


def rapport_fautes(nt, mode=0, actif=True):
    """'W' [mode, rapport] : mode 0 = pas de redondance, 1 = toujours, 2 = automatique ; rapport = lignes H et G."""
    import monitor_collect as mc
    mc._cmd(nt.target, "W", bytearray([int(mode), 1 if actif else 0]))


def operation(nt, pt):
    """Un chiffrement SANS oscilloscope : renvoie dict(ct, g, h, plog_hex, brut)."""
    import sca_capture as sc
    nt.target.simpleserial_write("p", bytearray(pt))
    buf = sc._read_until_z(nt.target)
    m = RE_R.search(buf)
    p = RE_P.search(buf)
    return dict(ct=bytes.fromhex(m.group(1)) if m else None, g=parse_g(buf), h=parse_h(buf),
                plog_hex=p.group(1) if p else None, brut=buf)


def signaux_normaux(nt, n=300, seed=11, key=None):
    """n chiffrements en usage normal avec la configuration déjà chargée : renvoie les lignes G et H."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pt = rng.integers(0, 256, 16, dtype=np.uint8)
        o = operation(nt, pt)
        o["pt"] = pt
        out.append(o)
    return out
