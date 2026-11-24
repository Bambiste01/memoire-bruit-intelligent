"""
evaluer_v10.py — section 6 du notebook en dehors de Jupyter (attaque CPA des campagnes v10, fenêtre (300, 24000)).

Pourquoi un script : dans un noyau Jupyter, numpy a déjà lancé ses fils de calcul (OpenBLAS) ; un « fork » des
processus de calcul peut alors se bloquer sans rien consommer. Ici, les fils sont limités à 1 AVANT d'importer numpy.

  cd ~/chipwhisperer/projet
  nohup python evaluer_v10.py > /media/khadim/2A53DBCE3700DFDA/trace/v10/log_evaluer_v10.txt 2>&1 &
  tail -f /media/khadim/2A53DBCE3700DFDA/trace/v10/log_evaluer_v10.txt

Reprise automatique : une campagne dont <campagne>_resultat_w300.json existe est sautée.
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

NPROC = int(os.environ.get("NPROC", 0)) or bv.nproc_sur()
B = json.load(open(os.path.join(bv.TRACE_ROOT, "base_v10.json")))
BASE_CYC, P_REF = B["BASE_CYC"], B["ref"]
CAMP = os.path.join(bv.TRACE_ROOT, "campagnes")
CAMPAGNES_V10 = {
    "sans protection": P_REF,
    "délais avant, moyenne flott.": CAMP + "/delais_avant_fm_20k",
    "délais avant, uniformes": CAMP + "/delais_avant_uniforme_20k",
    "mélange seul": CAMP + "/melange_20k",
    "niveau Très fort": CAMP + "/tres_fort_20k",
}

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
log("référence propre :", P_REF, "| processus :", NPROC)
clean = sm.clean_reference(P_REF, tuple(range(16)), window=bv.WIN)
res = {}
for nom, p in CAMPAGNES_V10.items():
    log("début :", nom)
    res[nom] = bv.evaluer(p, clean, nproc=NPROC)
    log("fin :", nom, "| X médian =", res[nom]["X_median"])
rows = bv.tableau(res, BASE_CYC)
json.dump(rows, open(os.path.join(bv.TRACE_ROOT, "tableau_v10.json"), "w"), indent=1)
log("FINI ->", os.path.join(bv.TRACE_ROOT, "tableau_v10.json"))
