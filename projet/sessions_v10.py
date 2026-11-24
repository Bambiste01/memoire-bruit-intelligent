"""
sessions_v10.py — étape 4.2 : sessions d'usage NORMAL réalistes (sans glitch), avec les événements légitimes
qu'un utilisateur produit : pauses irrégulières, resets (broche nRST), coupures d'alimentation, double reset.
Ce sont les données « normales » qui manquaient pour décider s'il faut réagir dès le premier reset.

Comme en production : après un reset, l'hôte recharge la clé, réactive le rapport de faute et la politique ; on
n'envoie JAMAIS 'R' (le budget de fuite doit survivre). Chaque opération et chaque événement sont enregistrés
dans <outdir>/normal_<nom>.jsonl (format compatible avec glitch_v10 : glitch = False).
"""
import json
import os
import time

import numpy as np

import fautes_v10 as fv

SCENARIOS = {
    "usage":          dict(n_ops=600, pause=(0.0, 0.3)),
    "reset_isole":    dict(n_ops=600, pause=(0.0, 0.2), reset_every=100),
    "alimentation":   dict(n_ops=600, pause=(0.0, 0.2), power_every=150),
    "double_reset":   dict(n_ops=600, pause=(0.0, 0.2), double_every=200),
}


def _rearmer(nt, key, mode_red=2):
    import monitor_collect as mc
    mc._flush(nt.target, 100)
    mc._cmd(nt.target, "k", bytearray(key))
    fv.rapport_fautes(nt, mode=mode_red)          # production : redondance automatique + rapport H/G
    mc._cmd(nt.target, "a", bytearray([1]))       # politique embarquée


def _reset_pin(nt):
    s = nt.scope
    s.io.nrst = "low"; time.sleep(0.05)
    s.io.nrst = "high_z"; time.sleep(0.25)


def _coupure(nt):
    s = nt.scope
    s.io.target_pwr = False; time.sleep(0.8)      # RAM effacée : démarrage à froid
    s.io.target_pwr = True; time.sleep(0.5)


def session(nt, key, nom, outdir, n_ops=600, pause=(0.0, 0.3), reset_every=None, power_every=None,
            double_every=None, seed=0, mode_red=2):
    path = os.path.join(outdir, "normal_%s.jsonl" % nom)
    if os.path.exists(path) and '"fin": true' in open(path).read():
        print("[%s] déjà enregistrée" % nom); return path
    os.makedirs(outdir, exist_ok=True)
    import sca_capture as sc
    rng = np.random.default_rng(seed)
    pts = rng.integers(0, 256, (n_ops, 16), dtype=np.uint8)
    refs = sc.expected_ct(bytes(key), pts)
    nt.scope.io.hs2 = "clkgen"                    # horloge propre : aucun glitch
    _rearmer(nt, key, mode_red)
    t0, faux = time.time(), 0
    with open(path, "w") as f:
        for i in range(n_ops):
            ev = None
            if reset_every and i and i % reset_every == 0:
                _reset_pin(nt); ev = "reset_pin"
            if power_every and i and i % power_every == 0:
                _coupure(nt); ev = "coupure_alim"
            if double_every and i and i % double_every == 0:
                _reset_pin(nt); _rearmer(nt, key, mode_red); fv.operation(nt, pts[i]); _reset_pin(nt); ev = "double_reset"
            if ev:
                _rearmer(nt, key, mode_red)
                f.write(json.dumps(dict(i=i, evenement=ev, t=round(time.time() - t0, 3))) + "\n")
            time.sleep(float(rng.uniform(*pause)))
            o = fv.operation(nt, pts[i])
            ok = o["ct"] is not None and o["ct"] == bytes(refs[i])
            faux += int(not ok)
            f.write(json.dumps(dict(i=i, glitch=False, classe="normal" if ok else "faute", g=o["g"], h=o["h"],
                                    plog=o["plog_hex"], t=round(time.time() - t0, 3))) + "\n")
        f.write(json.dumps(dict(fin=True, n_ops=n_ops, chiffres_faux=faux, duree_s=round(time.time() - t0))) + "\n")
    print("[%s] %d opérations, %d chiffrés faux, %.0f s" % (nom, n_ops, faux, time.time() - t0))
    return path


def resume(path):
    """Niveaux choisis par la politique, classes de décision et compteurs de faute observés pendant la session."""
    L = [json.loads(l) for l in open(path)]
    ops = [r for r in L if "g" in r and r.get("g")]
    evs = [r["evenement"] for r in L if "evenement" in r]
    plog = [bytes.fromhex(r["plog"]) for r in ops if r.get("plog")]
    niv = [p[0] for p in plog]
    x6 = [(p[15] >> 1) & 7 for p in plog]
    cls = [(p[15] >> 4) & 7 for p in plog]
    return dict(operations=len(ops), evenements=evs,
                niveau_3_pct=round(100 * np.mean([n == 2 for n in niv]), 1) if niv else None,
                ops_classe_faute=int(sum(c == 2 for c in cls)), x6_max=max(x6) if x6 else None,
                boots_max=max(r["h"]["boots_recent"] for r in ops), causes_reset=sorted({r["h"]["reset_cause"] for r in ops}),
                demarrage_a_froid=int(sum(r["h"]["cold_boot"] for r in ops)))
