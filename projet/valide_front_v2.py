"""
valide_front_v2.py — validation des points du front (étape A4), à lancer dans un TERMINAL (long : ~1 h par
campagne de 20 000 traces avec 10 processus).

  cd ~/chipwhisperer/projets2
  python valide_front_v2.py traces/val11 traces/comp20k/delais_avant_20k traces/comp20k/TresFort_A_20k \
         --clean traces/comp/sans_bruit --nproc 10

Arguments : dossiers de campagnes, ou un dossier contenant index.json (sortie de capture_validation).
Résultats : <sortie>/<nom>_val_win300.json (reprise automatique : les campagnes déjà traitées sont sautées).
"""
import argparse
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parego_v2 as pv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("campagnes", nargs="+")
ap.add_argument("--clean", default="traces/comp/sans_bruit")
ap.add_argument("--base-cyc", type=float, default=42000.0)
ap.add_argument("--win", default="300,24000")
ap.add_argument("--nproc", type=int, default=None)
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--out", default="traces/val11/resultats")
a = ap.parse_args()

win = tuple(int(x) for x in a.win.split(","))
paths = []
for c in a.campagnes:
    idx = os.path.join(c, "index.json")
    paths += list(json.load(open(idx)).values()) if os.path.exists(idx) else [c]
os.makedirs(a.out, exist_ok=True)
clean = pv.make_clean(a.clean, win)
for p in paths:
    dst = os.path.join(a.out, "%s_val_win%d.json" % (os.path.basename(p.rstrip("/")), win[0]))
    if os.path.exists(dst):
        pv.log("déjà fait :", dst)
        continue
    pv.log("validation", p)
    r = pv.validate_campaign(p, clean, a.base_cyc, window=win, trials=a.trials, nproc=a.nproc)
    json.dump(r, open(dst, "w"), indent=1, default=float)
    pv.log("%s : surcoût %+.1f %%, X_median = %s, effort médian = %s bits" % (
        r["name"], r["overhead_pct"], r["X_median"], ["%.1f" % e for e in r["effort_med"]]))
