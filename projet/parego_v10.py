"""
parego_v10.py — étapes A2 + A3 (ParEGO v2 sur le banc, firmware v10) en dehors de Jupyter.

Pourquoi un script : chaque point capturé est évalué par des processus parallèles ; dans Jupyter, un blocage de ces
processus ne se voit pas. Ici : fils de calcul limités, nombre de processus adapté à la mémoire, journal horodaté.

Avant de lancer : libérer la carte dans le notebook (nt.scope.dis(); nt.target.dis()) ou arrêter son noyau.
  cd ~/chipwhisperer/projet
  nohup python parego_v10.py > /media/khadim/2A53DBCE3700DFDA/trace/v10/parego/log_parego_v10.txt 2>&1 &
  tail -f /media/khadim/2A53DBCE3700DFDA/trace/v10/parego/log_parego_v10.txt
Reprise automatique : relancer la même commande après une interruption (historique JSON).
Durée estimée : ~40 points x (capture 6000 traces ~4 min + évaluation) -> plusieurs heures, puis A3.
"""
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, "/home/khadim/chipwhisperer/monitor"]

import json  # noqa: E402
import time  # noqa: E402

import banc_v10 as bv  # noqa: E402
import noise_opt as no  # noqa: E402
import parego_v2 as pv  # noqa: E402

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
bv.verifier_disque()
B = json.load(open(os.path.join(bv.TRACE_ROOT, "base_v10.json")))
C = json.load(open(os.path.join(bv.TRACE_ROOT, "calibration_v10.json")))
P_REF, BASE_CYC, R_DELAIS = B["ref"], C["BASE_CYC"], int(C["R_DELAIS"])
if C.get("build") and C["build"] != bv.empreinte_firmware():
    raise SystemExit("Le .hex actuel (%s) n'est pas le build de la référence (%s) : reprogrammer ou refaire la "
                     "référence." % (bv.empreinte_firmware(), C["build"]))

nt = bv.connecter(programmer=False)                 # le firmware v10 est déjà sur la carte
info = bv.info_firmware(nt)
if info["version"] != bv.FW_VERSION_ATTENDUE:
    raise SystemExit("La carte n'exécute pas le firmware v10.")

NPROC = int(os.environ.get("NPROC", 0)) or bv.nproc_sur()
OPT = bv.dossier("parego", "opt11")
HIST = os.path.join(OPT, "history.json")
log("référence :", P_REF, "| BASE_CYC = %.0f | délais calibrés = %d | processus = %d" % (BASE_CYC, R_DELAIS, NPROC))

clean = pv.make_clean(P_REF)
p_a1 = os.path.join(bv.TRACE_ROOT, "parego", "opt10_history_win300.json")
hist_a1 = json.load(open(p_a1)) if os.path.exists(p_a1) else None
DELAIS_AVANT = lambda r: no.Config(((1, 1), (4, 1)), ((1, int(r)),), 0x403)
seeds = pv.seed_configs("levels_3niv.json", rescored=hist_a1)
seeds = [c if c.name != pv.CFG_DELAIS_AVANT.name else DELAIS_AVANT(R_DELAIS) for c in seeds]
log("%d points de départ (A1 %s)" % (len(seeds), "inclus" if hist_a1 else "non disponible"))

# A2 — ParEGO
pv.run_parego_v2(nt, bv.KEY, clean, BASE_CYC, seeds=seeds, history_path=HIST,
                 n_init=6, n_iter=25, max_overhead=50, n_traces=6000, nproc=NPROC)
log("A2 terminé")

# A3 — seconde campagne des points du front (autres textes clairs), puis moyenne
hist_moy = pv.replicate_front(nt, bv.KEY, clean, BASE_CYC, history_path=HIST, n_traces=6000, nproc=NPROC)
for h in pv.front(hist_moy):
    log("front : %-80s %+6.1f %%  log10N = %.2f ± %.2f" % (h["config"]["name"][:80], h["overhead_pct"], h["log10_n"],
                                                          h.get("log10_n_sd", float("nan"))))
log("FINI ->", HIST.replace(".json", "_moy.json"))
