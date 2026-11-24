"""
valider_v10.py — étape A4 : attaque des campagnes de validation (20 000 traces) avec le MÊME attaquant que la
section 6 (verify_n, fenêtre (300, 24000)).

Organisation des campagnes (section 8 du notebook) :
  série s0 (graine 0) : trace/v10/validation/index.json      + références déjà attaquées en section 6
                        (campagnes/tres_fort_20k, campagnes/delais_avant_fm_20k)
  série s1 (graine 1) : trace/v10/validation/s1/index.json

Par défaut, seules les configurations qui protègent la ronde 10 (masque 0x403 ou 0x7FF) sont attaquées : la capture
ne voit pas la ronde 10, donc les configurations 0x003/0x007 paraissent artificiellement fortes. TOUT=1 les inclut.

  cd ~/chipwhisperer/projet
  nohup python valider_v10.py > /media/khadim/2A53DBCE3700DFDA/trace/v10/validation/log_validation.txt 2>&1 &

Reprise automatique : un fichier <campagne>_resultat_w300.json par campagne (les campagnes déjà traitées sont lues).
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

TRES_FORT = "permutation@subbytes_pre:4+jitter@mixcolumns:100+permutation@mixcolumns:4+permutation@addroundkey_pre:4|m403"
DELAIS_712 = "jitter@subbytes_pre:712+jitter@addroundkey_pre:712|m403"
CANDIDAT = "permutation@subbytes_pre:0+permutation@mixcolumns:0+jitter@addroundkey_pre:536+permutation@addroundkey_pre:0|m403"

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
TOUT = os.environ.get("TOUT") == "1"
VAL = os.path.join(bv.TRACE_ROOT, "validation")
CAMP = os.path.join(bv.TRACE_ROOT, "campagnes")


def protege_r10(nom):
    return bool(int(nom.rsplit("|m", 1)[1], 16) & 0x400)


camps = {"s0": {TRES_FORT: os.path.join(CAMP, "tres_fort_20k"), DELAIS_712: os.path.join(CAMP, "delais_avant_fm_20k")},
         "s1": {}}
for serie, d in (("s0", VAL), ("s1", os.path.join(VAL, "s1"))):
    idx = os.path.join(d, "index.json")
    if os.path.exists(idx):
        for nom, p in json.load(open(idx)).items():
            if TOUT or protege_r10(nom):
                camps[serie][nom] = p

# ordre : candidat principal d'abord, puis les autres ; série s0 puis s1
ordre = [(s, n) for s in ("s0", "s1") for n in sorted(camps[s], key=lambda n: (n != CANDIDAT, n))]
ordre = [(s, n) for s, n in ordre if os.path.exists(os.path.join(camps[s][n], "meta.json"))]
NPROC = int(os.environ.get("NPROC", 0)) or bv.nproc_sur()
log("%d campagnes à attaquer | %d processus | configurations 0x003/0x007 %s"
    % (len(ordre), NPROC, "incluses" if TOUT else "exclues"))

B = json.load(open(os.path.join(bv.TRACE_ROOT, "base_v10.json")))
clean = sm.clean_reference(B["ref"], tuple(range(16)), window=bv.WIN)
out_path = os.path.join(VAL, "resultats_validation.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for i, (serie, nom) in enumerate(ordre, 1):
    log("[%d/%d] début : %s | %s" % (i, len(ordre), serie, nom))
    r = bv.evaluer(camps[serie][nom], clean, nproc=NPROC)
    res.setdefault(serie, {})[nom] = r
    json.dump(res, open(out_path, "w"), indent=1)          # écrit après chaque campagne
    log("[%d/%d] fin : X médian = %s | %.0f cycles" % (i, len(ordre), r["X_median"], r["trig_mean"]))
log("FINI ->", out_path)
