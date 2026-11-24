"""
grande_v10.py — option C : attaque de la grande campagne (50 000 traces) du niveau 3, pour savoir si la cible
de 10⁴ traces (ISO/IEC 17825, niveau 3) est atteinte. Même attaquant que la section 6 et que valider_v10.py
(A1 à A4, fenêtre (300, 24000)), mais courbe prolongée jusqu'à 25 000 traces.

Mémoire : chaque processus réaligne 50 000 traces x ~8 000 échantillons (~3,5 Go) -> 3 processus par défaut.
  cd ~/chipwhisperer/projet
  nohup python grande_v10.py > /media/khadim/2A53DBCE3700DFDA/trace/v10/validation/grande/log_grande.txt 2>&1 &
Reprise : le résultat est écrit dans <campagne>_resultat_w300.json (relu s'il existe).
"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json  # noqa: E402
import time  # noqa: E402

import banc_v10 as bv  # noqa: E402
import sca_metrics as sm  # noqa: E402

NS = (20, 50, 100, 200, 300, 500, 800, 1200, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 16000, 20000, 25000)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
P = os.path.join(bv.TRACE_ROOT, "validation", "grande", "candidat_50k")
if not os.path.exists(os.path.join(P, "meta.json")):
    sys.exit("campagne absente ou incomplète : " + P)
NPROC = int(os.environ.get("NPROC", 3))
B = json.load(open(os.path.join(bv.TRACE_ROOT, "base_v10.json")))
clean = sm.clean_reference(B["ref"], tuple(range(16)), window=bv.WIN)
log("début :", P, "| %d processus" % NPROC)
r = bv.evaluer(P, clean, ns=NS, nproc=NPROC)
log("X médian = %s | bits restants à %d traces : médiane %.1f, p10 %.1f | octets trouvés %.1f"
    % (r["X_median"], r["ns"][-1], r["effort_med"][-1], r["effort_p10"][-1], r["octets"][-1]))
for n, e, p in zip(r["ns"], r["effort_med"], r["effort_p10"]):
    log("   n = %6d : effort médian %5.1f bits | p10 %5.1f" % (n, e, p))
log("FINI")
