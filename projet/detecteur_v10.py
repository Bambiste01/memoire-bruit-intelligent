"""
detecteur_v10.py — étape 4.3 : jeu de données et modèle appris de détection d'attaque par fautes (déployé sur la cible).

Données (toutes mesurées sur le banc, dossier trace/v10/fautes) :
  * campagnes de glitch d'horloge LARGES   horloge_{sans_redondance, redondance_auto, redondance_toujours}.jsonl
  * campagnes de glitch d'horloge CIBLÉES  cible_{...}.jsonl
    (blocs alternés : 50 opérations normales / 200 essais glitchés)
  * sessions d'usage NORMAL                normal_{usage, reset_isole, alimentation, double_reset}.jsonl

Chaque essai devient un événement : reset, ou opération (désaccord de redondance, erreur du compteur de rondes,
écart de durée nette). Un simulateur rejoue, à partir de ces événements, les compteurs entiers qu'un firmware peut
tenir (C, quelques octets en .noinit). Le simulateur est validé contre les compteurs réellement renvoyés par la
carte (fidelite()). Les caractéristiques par essai servent à entraîner un arbre de décision de faible profondeur,
exporté en C (export_c) pour la politique embarquée.

Étiquette : un essai est « attaque » s'il appartient à un bloc glitché ET qu'au moins un symptôme est déjà
apparu dans ce bloc (avant, l'essai est indécidable : il est exclu de l'apprentissage, mais compté dans le délai).
"""
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

NETREF = 47543            # durée nette de référence (cellule 9.5)
SEUIL_DUREE = 25          # cycles ; normal : max 15 sur 2400 opérations
FEATS = ["rs16", "rs64", "boots", "gap", "det", "rnd", "dur"]
DESCR = {"rs16": "resets récents (demi-vie 16 op.)", "rs64": "resets récents (demi-vie 64 op.)",
         "boots": "resets récents (demi-vie 256 op., = boots_recent v10.1)", "gap": "opérations depuis le dernier reset",
         "det": "désaccords de redondance récents", "rnd": "erreurs du compteur de rondes récentes",
         "dur": "durées nettes anormales récentes"}
LARGES = ["horloge_sans_redondance", "horloge_redondance_auto", "horloge_redondance_toujours"]
CIBLEES = ["cible_sans_redondance", "cible_redondance_auto", "cible_redondance_toujours"]
NORMALES = ["usage", "reset_isole", "alimentation", "double_reset"]


def _load(f):
    return [json.loads(l) for l in open(f)]


def charger(dossier):
    camps = {n: _load(os.path.join(dossier, n + ".jsonl")) for n in LARGES + CIBLEES}
    norm = {n: _load(os.path.join(dossier, "normal_%s.jsonl" % n)) for n in NORMALES}
    return camps, norm


# ---------------------------------------------------------------- flux d'événements
def ev_campagne(R):
    ev, b, prev = [], -1, None
    for r in R:
        if "glitch" not in r:
            continue
        if r["glitch"] != prev:
            b += 1; prev = r["glitch"]
        g = r["g"]
        if r["classe"] == "reset":
            e = dict(t="reset")
        else:
            e = dict(t="op", mis=bool(g["mismatch"]), ronde=bool(g["round_err_op"]), dnet=abs(g["net_cycles"] - NETREF))
        e.update(bloc=b, glitch=r["glitch"]); ev.append(e)
    return ev


def ev_normal(R):
    ev = []
    for r in R:
        if "evenement" in r:
            k = r["evenement"]
            if k == "reset_pin":
                new = [dict(t="reset")]
            elif k == "double_reset":          # reset, une opération, reset
                new = [dict(t="reset"), dict(t="op", mis=False, ronde=False, dnet=0, cache=True), dict(t="reset")]   # op. intermédiaire non enregistrée
            else:                              # coupure d'alimentation : RAM effacée
                new = [dict(t="froid")]
            for e in new:
                e.update(glitch=False, bloc=0)
            ev += new
        elif r.get("g"):
            g = r["g"]
            ev.append(dict(t="op", mis=bool(g["mismatch"]), ronde=bool(g["round_err_op"]),
                           dnet=abs(g["net_cycles"] - NETREF), glitch=False, bloc=0))
    return ev


# ---------------------------------------------------------------- compteurs embarqués (simulateur)
def simule(ev, frais_par_bloc=True, decroissance_persistante=True):
    """Une ligne par essai = état des compteurs après l'événement (sert à choisir le niveau de l'opération suivante).
    decroissance_persistante=False reproduit le firmware v10.1, où le compteur g_ops_since_decay est en RAM
    volatile : il repart de 0 à chaque reset, donc boots_recent ne décroît jamais si les resets sont espacés
    de moins de 256 opérations."""
    X, meta = [], []
    s, prevb, vu = None, None, False
    for e in ev:
        if s is None or e["t"] == "froid" or (frais_par_bloc and e["bloc"] != prevb):
            s = dict(rs16=0, rs64=0, boots=0, gap=255, det=0, rnd=0, dur=0, ops=0); prevb = e["bloc"]; vu = False
            if e["t"] == "froid":
                continue
        if e["t"] == "reset":
            s["rs16"] = min(15, s["rs16"] + 1); s["rs64"] = min(15, s["rs64"] + 1); s["boots"] = min(4, s["boots"] + 1)
            s["gap"] = 0; vu = True
            if not decroissance_persistante:
                s["ops"] = 0
        else:
            if e["mis"]:
                s["det"] = min(4, s["det"] + 1); vu = True
            if e["ronde"]:
                s["rnd"] = min(4, s["rnd"] + 1); vu = True
            if e["dnet"] > SEUIL_DUREE:
                s["dur"] = min(4, s["dur"] + 1); vu = True
            s["gap"] = min(255, s["gap"] + 1); s["ops"] = (s["ops"] + 1) & 0xFFFF
            if s["ops"] % 16 == 0:
                s["rs16"] >>= 1
            if s["ops"] % 64 == 0:
                s["rs64"] >>= 1
            if s["ops"] % 256 == 0:
                s["det"] >>= 1; s["rnd"] >>= 1; s["boots"] >>= 1; s["dur"] >>= 1
        X.append([s[k] for k in FEATS]); meta.append((e["bloc"], e["glitch"], e["glitch"] and vu))
    return np.array(X, dtype=int), meta


def fidelite(norm):
    """Accord (%) entre boots_recent mesuré par la carte et le simulateur en mode firmware v10.1."""
    out = {}
    for n, R in norm.items():
        ev = ev_normal(R)
        X, _ = simule(ev, False, decroissance_persistante=False)
        evs = [e for e in ev if e["t"] != "froid"]
        sim = np.array([X[k, FEATS.index("boots")] for k, e in enumerate(evs) if e["t"] == "op"])
        mes = np.array([r["h"]["boots_recent"] for r in R if r.get("h")])
        L = min(len(sim), len(mes))
        out[n] = round(100 * float(np.mean(sim[:L] == mes[:L])), 1)
    return out


# ---------------------------------------------------------------- jeux de données
def jeu(camps, norm, noms_c, noms_n, decroissance_persistante=True):
    Xs, y, lab, grp = [], [], [], []
    for n in noms_c:
        X, m = simule(ev_campagne(camps[n]), True, decroissance_persistante)
        Xs.append(X); y += [g for b, g, v in m]; lab += [v for b, g, v in m]; grp += [(n, b) for b, g, v in m]
    for n in noms_n:
        X, m = simule(ev_normal(norm[n]), False, decroissance_persistante)
        Xs.append(X); y += [False] * len(m); lab += [False] * len(m); grp += [("normal_" + n, 0)] * len(m)
    return dict(X=np.vstack(Xs), y=np.array(y), lab=np.array(lab), grp=grp)


def poids(J, k_fa=1.0):
    """Coût d'une fausse alerte en usage normal = k_fa fois celui d'un essai d'attaque manqué (la redondance double
    la durée). Chaque source normale pèse autant que l'ensemble des essais d'attaque."""
    lab, y = J["lab"], J["y"]
    src = np.array([a for a, b in J["grp"]])
    w = np.ones(len(lab)); npos = lab.sum()
    for g in np.unique(src):
        m = (src == g) & ~lab
        if m.any():
            w[m] = npos / m.sum() * (k_fa if g.startswith("normal_") else 1.0)
    w[y & ~lab] = 0.0
    return w


# Contraintes de monotonie (connaissance physique) : plus de resets / désaccords / erreurs => jamais moins d'alerte ;
# plus d'opérations depuis le dernier reset => jamais plus d'alerte. Évite qu'un arbre appris sur peu de données
# normales apprenne des règles incohérentes (ex. « 4 resets mais aucun récent => alerte, un récent => pas d'alerte »).
MONOTONIE = dict(rs16=1, rs64=1, boots=1, gap=-1, det=1, rnd=1, dur=1)


def entrainer(J, profondeur=3, k_fa=1.0, min_feuille=20, monotone=True):
    from sklearn.tree import DecisionTreeClassifier
    cst = [MONOTONIE[k] for k in FEATS] if monotone else None
    return DecisionTreeClassifier(max_depth=profondeur, min_samples_leaf=min_feuille, random_state=0,
                                  monotonic_cst=cst).fit(
        J["X"], J["lab"], sample_weight=poids(J, k_fa))


def regle_x6(X):
    f = dict(zip(FEATS, X.T))
    return (f["det"] + f["rnd"] + np.maximum(0, f["boots"] - 1)) > 0


def preuve_faute(X):
    """Symptômes qui prouvent une faute (jamais observés en usage normal : 0 sur 2400 opérations) :
    désaccord de redondance, erreur du compteur de rondes, durée nette anormale. Règle de sûreté conservée telle
    quelle ; le modèle appris décide seulement des cas ambigus (resets)."""
    f = dict(zip(FEATS, X.T))
    return (f["det"] + f["rnd"] + f["dur"]) > 0


def decision(arbre, X):
    """Décision déployée : règle de sûreté OU arbre appris."""
    return preuve_faute(X) | arbre.predict(X).astype(bool)


def maintien(pred, grp, hold):
    """Hystérésis : une alerte maintient le niveau 3 pendant `hold` opérations."""
    out = pred.copy(); c = 0; prev = None
    for k, gb in enumerate(grp):
        if gb != prev:
            c = 0; prev = gb
        if pred[k]:
            c = hold
        elif c > 0:
            out[k] = True; c -= 1
    return out


def evaluer(pred, J):
    df = pd.DataFrame(dict(p=pred, y=J["y"], g=[a for a, b in J["grp"]], b=[b for a, b in J["grp"]]))
    gl = df[df.y]
    delais = []
    for _, d in gl.groupby(["g", "b"]):
        idx = np.flatnonzero(d.p.values); delais.append(idx[0] + 1 if len(idx) else np.inf)
    delais = np.array(delais, dtype=float); fin = np.isfinite(delais)
    out = {"blocs détectés": "%d/%d" % (fin.sum(), len(delais)),
           "délai médian (essais)": float(np.median(delais)),
           "délai p90": float(np.percentile(delais, 90)) if fin.all() else np.inf,
           "couverture attaque %": round(100 * gl.p.mean(), 1),
           "FA blocs normaux campagne %": round(100 * df[(~df.y) & (~df.g.str.startswith("normal_"))].p.mean(), 2)}
    for g, d in df[df.g.str.startswith("normal_")].groupby("g"):
        out["FA %s %%" % g[7:]] = round(100 * d.p.mean(), 1)
    return out


SPLITS = {
    "A": dict(train=(LARGES, ["usage", "reset_isole"]), test=(CIBLEES, ["double_reset", "alimentation"])),
    "B": dict(train=(CIBLEES, ["double_reset", "alimentation"]), test=(LARGES, ["usage", "reset_isole"])),
}


def validation_croisee(camps, norm, profondeur=3, k_fa=1.0, hold=0, monotone=True):
    """Entraînement sur une famille de glitch + deux scénarios normaux ; test sur l'autre famille + les deux autres
    scénarios (jamais vus). Comparaison à la règle x6 du firmware v10.1, rejouée exactement."""
    rows = []
    for s, d in SPLITS.items():
        Jtr = jeu(camps, norm, *d["train"]); Jte = jeu(camps, norm, *d["test"])
        Jv = jeu(camps, norm, *d["test"], decroissance_persistante=False)
        t = entrainer(Jtr, profondeur, k_fa, monotone=monotone)
        rows.append(dict(split=s, détecteur="règle x6 (firmware v10.1)", **evaluer(maintien(regle_x6(Jv["X"]), Jv["grp"], 0), Jv)))
        rows.append(dict(split=s, détecteur="règle x6, décroissance corrigée", **evaluer(regle_x6(Jte["X"]), Jte)))
        rows.append(dict(split=s, détecteur="sûreté + arbre appris (p%d, kFA %g, maintien %d%s)" % (profondeur, k_fa, hold, ", monotone" if monotone else ""),
                         **evaluer(maintien(decision(t, Jte["X"]), Jte["grp"], hold), Jte)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- export C
def export_c(arbre, chemin, hold=0, commentaire=""):
    from sklearn.tree import _tree
    T = arbre.tree_
    lignes = []

    def rec(n, ind):
        sp = "    " * ind
        if T.feature[n] == _tree.TREE_UNDEFINED:
            v = T.value[n][0]; lignes.append(sp + "return %d;" % int(v[1] > v[0])); return
        f = FEATS[T.feature[n]]; thr = int(np.floor(T.threshold[n]))      # caractéristiques entières : x <= floor(s)
        lignes.append(sp + "if (f->%s <= %d) {" % (f, thr)); rec(T.children_left[n], ind + 1)
        lignes.append(sp + "} else {"); rec(T.children_right[n], ind + 1); lignes.append(sp + "}")

    lignes.append("    if (f->det | f->rnd | f->dur) return 1;   /* règle de sûreté : preuve de faute */")
    rec(0, 1)
    h = ["/* fault_tree.h — généré par detecteur_v10.export_c ; NE PAS ÉDITER À LA MAIN. */",
         "/* " + commentaire + " */", "#ifndef FAULT_TREE_H", "#define FAULT_TREE_H", "#include <stdint.h>", "",
         "#define FAULT_TREE_HOLD %d   /* opérations maintenues au niveau 3 après une alerte */" % hold, "",
         "typedef struct { uint8_t rs16, rs64, boots, gap, det, rnd, dur; } fault_feat_t;", "",
         "static inline uint8_t fault_tree_alarm(const fault_feat_t *f)", "{"] + lignes + ["}", "", "#endif", ""]
    with open(chemin, "w") as fh:
        fh.write("\n".join(h))
    return chemin


def sensibilite(camps, norm, profondeurs=(2, 3, 4), k_fas=(1, 3, 10), holds=(0, 32), monotones=(True, False)):
    """Annexe : sensibilité aux hyperparamètres (texte : p3, kFA 1, maintien 0, monotone)."""
    rows = []
    for s, d in SPLITS.items():
        Jtr = jeu(camps, norm, *d["train"]); Jte = jeu(camps, norm, *d["test"])
        for mo in monotones:
          for p in profondeurs:
            for k in k_fas:
                t = entrainer(Jtr, p, k, monotone=mo)
                for h in holds:
                    o = evaluer(maintien(decision(t, Jte["X"]), Jte["grp"], h), Jte)
                    fa = [v for kk, v in o.items() if kk.startswith("FA ") and "campagne" not in kk]
                    rows.append(dict(split=s, monotone=mo, profondeur=p, k_fa=k, maintien=h, detectes=o["blocs détectés"],
                                     delai_median=o["délai médian (essais)"], couverture=o["couverture attaque %"],
                                     FA_max_normal=max(fa)))
    return pd.DataFrame(rows)


# ================================================================ ARBRE UNIQUE DE POLITIQUE (même logique que la v9)
# Un seul arbre exporté dans noise_policy.h. Ordre des nœuds :
#   (banc) limite = 0                                   -> niveau 3, classe 3
#   1. règle de sûreté : faute prouvée (det | rnd | dur) -> niveau 3, classe 2   (fixée : non apprenable, voir texte)
#   2. sous-arbre APPRIS sur les resets (monotone)       -> niveau 3, classe 2
#   3. nœud budget DISTILLÉ du planificateur (x1)        -> niveau 3, classe 1
#   4. sinon plancher calibré (tirage a/b)               -> classe 0
# Classe 2 = faute : le firmware v12 active aussi la redondance.

def distiller_budget(n=20000, x1_max=1300, seed=0):
    """Distillation du planificateur de budget (comme policy_train.py en v9) : on tire des états de budget x1 (‰),
    le planificateur étiquette « épuisé » si x1 >= 1000, et un arbre de profondeur 1 apprend le seuil."""
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, x1_max + 1, n)
    t = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x1.reshape(-1, 1), x1 >= 1000)
    seuil = int(np.floor(t.tree_.threshold[0]))          # x1 <= seuil -> non épuisé
    return dict(arbre=t, seuil=seuil, accord=float(np.mean(t.predict(x1.reshape(-1, 1)) == (x1 >= 1000))))


def construire_arbre_unique(J, profondeur=3, k_fa=1.0, seed=0):
    """Assemble l'arbre unique : sous-arbre des resets appris sur J (jeu de données réel) + nœud budget distillé."""
    return dict(resets=entrainer(J, profondeur, k_fa), budget=distiller_budget(seed=seed),
                profondeur=profondeur, k_fa=k_fa)


def decide_unique_py(AU, x1, X, limit=1, r16=None, cfg=None):
    """Réplique Python de policy_decide v12. X : caractéristiques de faute (n, 7) ; x1 : budget (‰) (n,).
    Renvoie (niveau, classe) ; niveau = -1 pour le plancher si r16 est None (« a ou b »)."""
    X = np.atleast_2d(X); x1 = np.atleast_1d(x1)
    cls = np.zeros(len(X), dtype=int)
    if limit == 0:
        return np.full(len(X), 2), np.full(len(X), 3)
    faute = preuve_faute(X) | AU["resets"].predict(X).astype(bool)
    cls[faute] = 2
    cls[(~faute) & (x1 > AU["budget"]["seuil"])] = 1
    niv = np.where(cls > 0, 2, -1)
    if r16 is not None:
        r16 = np.atleast_1d(r16)
        niv = np.where(cls == 0, np.where(r16 < cfg["p16"], cfg["a"], cfg["b"]), niv)
    return niv, cls


def _c_sous_arbre(arbre, ind):
    from sklearn.tree import _tree
    T = arbre.tree_
    L = []

    def rec(n, k):
        sp = "    " * k
        if T.feature[n] == _tree.TREE_UNDEFINED:
            v = T.value[n][0]
            if v[1] > v[0]:
                L.append(sp + "return 1;   /* resets suspects (appris) */")
            else:
                L.append(sp + "return 0;   /* resets compatibles avec un usage normal */")
            return
        f = FEATS[T.feature[n]]; thr = int(np.floor(T.threshold[n]))
        L.append(sp + "if (f->%s <= %d) {" % (f, thr)); rec(T.children_left[n], k + 1)
        L.append(sp + "} else {"); rec(T.children_right[n], k + 1); L.append(sp + "}")

    rec(0, ind)
    return L


def export_politique(AU, niv, n_vie=10000, m=2, chemin=None, commentaire=""):
    """Génère noise_policy.h v12 (arbre unique). Reprend la table des niveaux et le plancher de
    politique_v10.export_header (appelé avec path=None : rien n'est écrit dans le firmware), et remplace
    policy_decide par l'arbre unique. Par défaut, écrit dans projet/ (PAS dans le dossier du firmware)."""
    import politique_v10 as pv
    base = pv.export_header(niv, n_vie=n_vie, m=m, path=None)
    h = base["header"]
    i0 = h.index("static uint8_t policy_decide(")
    i1 = h.index("#endif /* NOISE_POLICY_H_ */")
    corps = ["typedef struct { uint8_t rs16, rs64, boots, gap, det, rnd, dur; } fault_feat_t;   /* compteurs de faute */",
             "",
             "/* Arbre unique (v12), branche « faute » : nœuds 1 et 2. Utilisée par policy_decide ET par la redondance",
             "   automatique (mode 2), pour que la redondance suive exactement la classe 2 de l'arbre. Fonction pure. */",
             "static inline uint8_t policy_fault_alarm(const fault_feat_t *f)",
             "{",
             "    if (f->det | f->rnd | f->dur) return 1;   /* 1. sûreté : faute prouvée */",
             "    /* 2. sous-arbre appris sur les resets (monotone, profondeur %d, kFA %g) */" % (AU["profondeur"], AU["k_fa"])]
    corps += _c_sous_arbre(AU["resets"], 1)
    corps += ["}",
              "",
              "/* Arbre unique (v12). x[1] = budget consommé (‰). */",
              "static uint8_t policy_decide(const uint16_t *x, const fault_feat_t *f, uint32_t leak_q16, uint32_t limit_q16,",
              "                             uint8_t *cls)",
              "{",
              "    (void)leak_q16;",
              "    if (limit_q16 == 0u) { *cls = 3; return POLICY_STRONGEST; }        /* budget désactivé (banc)          */",
              "    if (policy_fault_alarm(f)) { *cls = 2; return POLICY_STRONGEST; }   /* 1-2. faute : niveau 3 + redondance */"]
    corps += ["    if (x[1] > %du) { *cls = 1; return POLICY_STRONGEST; }   /* 3. budget épuisé (nœud distillé du planificateur) */"
              % AU["budget"]["seuil"],
              "    *cls = 0;                                                          /* 4. plancher calibré (mélange)    */",
              "    {",
              "        uint16_t r = (uint16_t)(((uint16_t)noise_rand8() << 8) | noise_rand8());",
              "        return (r < POLICY_MIX_P16) ? POLICY_MIX_A : POLICY_MIX_B;",
              "    }",
              "}", "", ""]
    h = h[:i0] + "\n".join(corps) + h[i1:]
    h = h.replace("FICHIER GÉNÉRÉ par politique_v10.export_header()  (firmware v10.1)",
                  "FICHIER GÉNÉRÉ par detecteur_v10.export_politique()  (firmware v12, arbre unique)")
    h = h.replace("Règles : faute détectée (x6 > 0) ou budget épuisé -> niveau",
                  "Arbre unique : sûreté (faute prouvée) / resets appris / budget distillé -> niveau")
    if commentaire:
        h = h.replace("*/\n#ifndef NOISE_POLICY_H_", "    " + commentaire + "\n*/\n#ifndef NOISE_POLICY_H_", 1)
    if chemin is None:
        chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_policy_v12.h")
    if "firmware" in os.path.abspath(chemin):
        raise ValueError("export_politique n'écrit pas dans le dossier du firmware sans accord explicite")
    with open(chemin, "w") as fh:
        fh.write(h)
    return dict(chemin=chemin, cfg=dict(a=base["a"], b=base["b"], p16=base["p16"]), header=h)


def verifier_c(AU, export, J, n_budget=(0, 500, 999, 1000, 1001, 1300)):
    """Compile policy_decide v12 sur le PC (gcc) avec un noise_rand8 bouchon et compare les classes C et Python sur
    tous les états de faute observés x quelques valeurs de budget. Renvoie (accords, total)."""
    import subprocess
    import tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "noise_generator.h"), "w").write(
        "#include <stdint.h>\n#define NOISE_TYPE_COUNT 5\nstatic uint8_t noise_rand8(void){return 0;}\n")
    open(os.path.join(d, "noise_policy.h"), "w").write(export["header"])
    U = np.unique(J["X"], axis=0)
    rows = []
    for b in n_budget:
        _, c = decide_unique_py(AU, np.full(len(U), b), U)
        rows += [list(u) + [b, cc] for u, cc in zip(U, c)]
    np.savetxt(os.path.join(d, "U.txt"), np.array(rows), fmt="%d")
    src = r'''#include <stdio.h>
#include "noise_policy.h"
int main(void){ int a[9], bad = 0, n = 0; FILE *u = fopen("%s/U.txt", "r");
  while (fscanf(u, "%%d %%d %%d %%d %%d %%d %%d %%d %%d", a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8) == 9) {
    fault_feat_t f = { a[0],a[1],a[2],a[3],a[4],a[5],a[6] }; uint16_t x[7] = {0}; uint8_t cls;
    x[1] = (uint16_t)a[7]; policy_decide(x, &f, 0, 1, &cls); bad += (cls != a[8]); n++; }
  printf("%%d %%d\n", n - bad, n); return 0; }''' % d
    open(os.path.join(d, "t.c"), "w").write(src)
    subprocess.run(["gcc", "-O2", "-Wall", "-I" + d, os.path.join(d, "t.c"), "-o", os.path.join(d, "t")], check=True,
                   capture_output=True)
    ok, n = map(int, subprocess.run([os.path.join(d, "t")], capture_output=True, text=True).stdout.split())
    return ok, n


def validation_arbre_unique(camps, norm, profondeur=3, k_fa=1.0, seed=0):
    """Même validation croisée que validation_croisee(), mais avec l'arbre unique complet (budget tiré au hasard
    dans [0, 1300] ‰ pour chaque essai de test) : vérifie que la partie faute garde ses performances et que la
    décision « niveau 3 » est correcte."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, d in SPLITS.items():
        Jtr = jeu(camps, norm, *d["train"]); Jte = jeu(camps, norm, *d["test"])
        Jv = jeu(camps, norm, *d["test"], decroissance_persistante=False)
        AU = construire_arbre_unique(Jtr, profondeur, k_fa, seed)
        x1 = rng.integers(0, 1301, len(Jte["X"]))
        _, c = decide_unique_py(AU, x1, Jte["X"])
        hors_faute = c != 2                     # là où l'arbre ne voit pas de faute, la classe 1 doit valoir « x1 >= 1000 »
        ok_budget = np.mean((c[hors_faute] == 1) == (x1[hors_faute] >= 1000))
        rows.append(dict(split=s, détecteur="règle x6 (firmware v10.1)", **evaluer(regle_x6(Jv["X"]), Jv)))
        rows.append(dict(split=s, détecteur="arbre unique (classe 2)", **evaluer(c == 2, Jte),
                         **{"budget : décision correcte %": round(100 * float(ok_budget), 2),
                            "seuil budget appris (‰)": AU["budget"]["seuil"]}))
    return pd.DataFrame(rows)


# ================================================================ VALIDATION v12 SUR LA CARTE
def lire_plog_v12(plog):
    """Journal 'P' v12 (16 octets) : [0] niveau, [1-2] x0, [3-4] x1 budget ‰, [5..11] rs16 rs64 boots gap det rnd dur
    (compteurs AU MOMENT de la décision), [13-14] cycles de décision, [15] bit0 épuisé, bit1 alerte faute, bits 4-6 classe."""
    b = bytes.fromhex(plog) if isinstance(plog, str) else bytes(int(v) for v in plog)
    return dict(level=b[0], x0=(b[1] << 8) | b[2], x1=(b[3] << 8) | b[4], feats=list(b[5:12]),
                decide_cycles=(b[13] << 8) | b[14], epuise=b[15] & 1, alerte=(b[15] >> 1) & 1, classe=(b[15] >> 4) & 7)


def conformite_v12(plogs, AU, cfg, alpha_z=4.0):
    """Rejoue l'arbre unique (Python) sur les compteurs lus par la carte et compare à sa décision.
    Décisions déterministes (classes 1, 2) : 100 % attendu. Plancher : niveau dans {a, b}, part de a compatible avec p."""
    bad, det, n_pl, n_a, ex = 0, 0, 0, 0, []
    for pl in plogs:
        d = lire_plog_v12(pl)
        X = np.array([d["feats"]])
        _, c = decide_unique_py(AU, np.array([d["x1"]]), X)
        c = int(c[0])
        if c != 0:
            det += 1
            ok = d["level"] == 2 and d["classe"] == c and d["alerte"] == int(c == 2)
        else:
            n_pl += 1; n_a += int(d["level"] == cfg["a"])
            ok = d["level"] in (cfg["a"], cfg["b"]) and d["classe"] == 0 and d["alerte"] == 0
        if not ok:
            bad += 1
            if len(ex) < 5:
                ex.append(dict(carte=d, python_classe=c))
    p = cfg["p16"] / 65536.0
    z = None
    if n_pl:
        sd = np.sqrt(p * (1 - p) / n_pl)
        z = (n_a / n_pl - p) / sd if sd > 0 else 0.0
    return dict(n=len(plogs), erreurs=bad, deterministes=det, plancher=n_pl, part_a=(n_a / n_pl if n_pl else None),
                p_attendu=p, z=z, conforme=(bad == 0 and (z is None or abs(z) < alpha_z)), exemples=ex)


def fidelite_v12(path):
    """Fichier .jsonl enregistré avec le firmware v12 (session normale ou campagne de glitch) : compare les compteurs
    lus par la carte au moment de chaque décision (journal 'P') à ceux du simulateur rejoué sur les mêmes événements.
    Le simulateur ne voit pas les remises à zéro du banc ('H' [1]) : on part de compteurs à zéro, comme les cellules."""
    R = _load(path)
    camp = any("width" in r for r in R)
    ev = ev_campagne(R) if camp else ev_normal(R)
    X, _ = simule(ev, frais_par_bloc=False, decroissance_persistante=True)
    evs = [e for e in ev if e["t"] != "froid"]
    # état utilisé pour décider l'opération k = état après l'événement précédent
    etats, prev = [], np.array([0, 0, 0, 255, 0, 0, 0])
    for k, e in enumerate(evs):
        if e["t"] == "op" and not e.get("cache"):      # l'op. intermédiaire du double reset n'est pas dans le fichier
            etats.append(prev)
        prev = X[k]
    ops = [r for r in R if r.get("g")]                                 # une entrée par opération AES
    mes = [lire_plog_v12(r["plog"])["feats"] if r.get("plog") else [-1] * 7 for r in ops]
    L = min(len(mes), len(etats))
    mes, sim = np.array(mes[:L]), np.array(etats[:L])
    acc = float(np.mean((mes == sim).all(axis=1))) if L else float("nan")
    return dict(operations=L, accord_pct=round(100 * acc, 2),
                premier_ecart=(None if acc == 1 else int(np.flatnonzero(~(mes == sim).all(axis=1))[0])))


def entrainement_detaille(J, profondeur=3, k_fa=1.0, seed=0, ax=None):
    """Affiche pas à pas l'entraînement de l'arbre unique (cellule 12.0) et renvoie AU."""
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import plot_tree
    src = np.array([a for a, b in J["grp"]])
    print("1) JEU DE DONNÉES : %d essais mesurés sur la carte" % len(J["y"]))
    for g in sorted(set(s.split("_")[0] for s in src)):
        m = np.array([s.startswith(g) for s in src])
        print("   %-8s : %6d essais | attaque observable %6d | normaux %6d" % (g, m.sum(), (J["lab"] & m).sum(), (~J["y"] & m).sum()))
    print("   caractéristiques :", ", ".join(FEATS))
    w = poids(J, k_fa)
    print("2) ENTRAÎNEMENT : DecisionTreeClassifier(profondeur %d, monotone, coût fausse alerte x%g) sur %d essais pondérés"
          % (profondeur, k_fa, int((w > 0).sum())))
    AU = construire_arbre_unique(J, profondeur, k_fa, seed)
    t = AU["resets"]
    m = w > 0
    pred = t.predict(J["X"][m]).astype(bool) | preuve_faute(J["X"][m])
    cm = confusion_matrix(J["lab"][m], pred)
    print("   matrice de confusion (apprentissage) [normal, attaque] x [prédit normal, prédit attaque] :\n  ", cm.tolist())
    print("   importance des caractéristiques :",
          {f: round(float(v), 3) for f, v in zip(FEATS, t.feature_importances_) if v > 0})
    print("3) NŒUD BUDGET distillé du planificateur : x1 <= %d ‰ -> non épuisé (accord %.2f %%)"
          % (AU["budget"]["seuil"], 100 * AU["budget"]["accord"]))
    if ax is not None:
        plot_tree(t, feature_names=FEATS, class_names=["normal", "attaque"], filled=True, impurity=False,
                  proportion=True, rounded=True, fontsize=7, ax=ax)
    return AU


def simulation_escalier(niv, n_vie=10000, m=2, p16=3121, a=1, b=2):
    """Étape 4.1 : budget de fuite cumulé L(k) = somme des 1/N du niveau joué, sur la vie de la clé.
    Trois ordonnancements :
      - mélange constant (firmware v10.1/v12) : niveau a avec la probabilité p, sinon b, tiré à chaque chiffrement ;
      - escalier, mêmes effectifs : les n_a chiffrements au niveau a d'abord, puis b (même coût, même fuite totale) ;
      - escalier glouton (logique v9) : niveau le moins cher tant que le budget 1/m n'est pas épuisé, puis niveau 3.
    Renvoie les courbes et le résumé (coût moyen, fuite finale, marge réelle = 1/L_final)."""
    N = [v["N"] for v in niv]; c = [v["surcout"] for v in niv]
    p = p16 / 65536.0; k = np.arange(1, n_vie + 1)
    rng = np.random.default_rng(0)
    lv_mel = np.where(rng.random(n_vie) < p, a, b)
    na = int(round(p * n_vie)); lv_esc = np.r_[np.full(na, a), np.full(n_vie - na, b)]
    lv_glo, L = [], 0.0
    for _ in range(n_vie):
        l = 0 if L < 1.0 / m else len(niv) - 1
        lv_glo.append(l); L += 1.0 / N[l]
    out = {}
    for nom, lv in (("mélange constant", lv_mel), ("escalier (mêmes effectifs)", lv_esc), ("escalier glouton (v9)", np.array(lv_glo))):
        Lk = np.cumsum([1.0 / N[l] for l in lv])
        out[nom] = dict(L=Lk, cout=float(np.mean([c[l] for l in lv])), L_final=float(Lk[-1]), marge=float(1 / Lk[-1]),
                        L_1000=float(Lk[999]))
    return k, out


# ================================================================ v13 : ESCALIER CALIBRÉ (SCA dynamique)
# Même famille de niveaux, même budget (limite 2^30/m) et même branche « faute » que la v12.
# Seul change le dernier nœud : au lieu du tirage constant a/b, la marche dépend du budget déjà consommé x1 :
#   niveau a (bon marché) tant que x1 <= seuil de marche, puis niveau b, puis « budget épuisé » (niveau 3 + alerte).
# Le seuil est calculé par le planificateur en rejouant l'arithmétique ENTIÈRE de la carte, puis distillé dans l'arbre.

def _fuites_q(header):
    """Fuite par chiffrement de chaque niveau (unité 2^30/N), lue dans la table générée (identique au firmware)."""
    import re
    return [int(v) for v in re.findall(r"0x[0-9A-Fa-f]+u, (\d+)u \}", header)]


def simule_budget_carte(niveau_de_x1, fuites, limite, n):
    """Rejoue noise_end_encryption + compute_features : x1 = (fuite * 1000) // limite, niveau choisi, fuite += f[niveau].
    Renvoie les niveaux joués et x1 avant chaque chiffrement."""
    L, lv, xs = 0, np.empty(n, int), np.empty(n, int)
    for i in range(n):
        x1 = min((L * 1000) // limite, 0xFFFF)
        l = niveau_de_x1(x1)
        lv[i], xs[i] = l, x1
        L = min(L + fuites[l], 0xFFFFFFFF)
    return lv, xs


def planificateur_escalier(niv, n_vie=10000, m=2):
    """Planificateur hors ligne : plus grand seuil de marche T (‰) tel que la règle
    « niveau a si x1 <= T, niveau b si T < x1 <= 999, niveau 3 au-delà » ne dépasse pas la limite avant n_vie
    chiffrements (arithmétique entière de la carte). a, b = les deux niveaux du plancher calibré (cellule 8.7)."""
    import politique_v10 as pv
    base = pv.export_header(niv, n_vie=n_vie, m=m, path=None)
    a, b, lim = base["a"], base["b"], base["limit"]
    f = _fuites_q(base["header"])
    top = len(niv) - 1
    def regle(T):
        return lambda x1: (top if x1 >= 1000 else (a if x1 <= T else b))
    def tient(T):                       # la fuite cumulée des n_vie chiffrements reste <= limite
        lv, _ = simule_budget_carte(regle(T), f, lim, n_vie)
        return sum(f[l] for l in lv) <= lim
    if not tient(-1):
        raise ValueError("aucun escalier ne tient n_vie = %d avec m = %d" % (n_vie, m))
    lo, hi = -1, 999                    # tient(T) est monotone : plus T est grand, plus on joue le niveau a
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if tient(mid) else (lo, mid - 1)
    best = lo
    lv, xs = simule_budget_carte(regle(best), f, lim, n_vie)
    return dict(a=a, b=b, seuil=best, limite=lim, fuites=f, n_a=int((lv == a).sum()), n_vie=n_vie, m=m,
                fuite_finale=float(sum(f[l] for l in lv)) / lim, base=base)


def distiller_escalier(plan, n=20000, x1_max=1300, seed=0):
    """Distillation (comme distiller_budget) : états x1 tirés au hasard, étiquetés par le planificateur
    (0 = marche a, 1 = marche b, 2 = épuisé) ; un arbre de profondeur 2 apprend les deux seuils."""
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, x1_max + 1, n)
    y = np.where(x1 >= 1000, 2, np.where(x1 <= plan["seuil"], 0, 1))
    t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x1.reshape(-1, 1), y)
    thr = sorted(int(np.floor(v)) for v, ft in zip(t.tree_.threshold, t.tree_.feature) if ft >= 0)
    acc = float(np.mean(t.predict(x1.reshape(-1, 1)) == y))
    grille = np.arange(0, x1_max + 1)
    yg = np.where(grille >= 1000, 2, np.where(grille <= plan["seuil"], 0, 1))
    acc_grille = float(np.mean(t.predict(grille.reshape(-1, 1)) == yg))
    return dict(arbre=t, seuil_marche=thr[0], seuil=thr[1], accord=acc, accord_grille=acc_grille)


def construire_arbre_v13(J, niv, n_vie=10000, m=2, profondeur=3, k_fa=1.0, seed=0):
    """Arbre unique v13 : sous-arbre des resets appris sur J (inchangé) + nœuds budget/escalier distillés."""
    plan = planificateur_escalier(niv, n_vie, m)
    esc = distiller_escalier(plan, seed=seed)
    return dict(resets=entrainer(J, profondeur, k_fa), budget=dict(seuil=esc["seuil"], accord=esc["accord"]),
                escalier=esc, plan=plan, profondeur=profondeur, k_fa=k_fa)


def decide_v13_py(AU, x1, X, limit=1):
    """Réplique Python de policy_decide v13 : renvoie (niveau, classe), entièrement déterministe."""
    X = np.atleast_2d(X); x1 = np.atleast_1d(x1)
    top = len(AU["plan"]["fuites"]) - 1
    if limit == 0:
        return np.full(len(X), top), np.full(len(X), 3)
    faute = preuve_faute(X) | AU["resets"].predict(X).astype(bool)
    cls = np.zeros(len(X), dtype=int)
    cls[faute] = 2
    cls[(~faute) & (x1 > AU["budget"]["seuil"])] = 1
    marche = np.where(x1 <= AU["escalier"]["seuil_marche"], AU["plan"]["a"], AU["plan"]["b"])
    return np.where(cls > 0, top, marche), cls


def export_politique_v13(AU, niv, chemin, commentaire=""):
    """Génère noise_policy.h v13 : en-tête v12 (mêmes niveaux, même branche faute) + escalier distillé."""
    if "firmware" in os.path.abspath(chemin):
        raise ValueError("export_politique_v13 n'écrit pas dans le dossier du firmware sans accord explicite")
    p, e = AU["plan"], AU["escalier"]
    tmp = chemin + ".v12tmp"
    h = export_politique(AU, niv, n_vie=p["n_vie"], m=p["m"], chemin=tmp, commentaire=commentaire)["header"]
    os.remove(tmp)
    cout_court = niv[p["a"]]["surcout"]
    cout_moy = (p["n_a"] * niv[p["a"]]["surcout"] + (p["n_vie"] - p["n_a"]) * niv[p["b"]]["surcout"]) / p["n_vie"]
    i0 = h.index("    Plancher calibré :"); i1 = h.index("\n", i0)
    h = h[:i0] + ("    Escalier calibré (v13) : niveau %d tant que x1 <= %d ‰ (%d premiers chiffrements), puis niveau %d ;"
                  " budget épuisé au-delà de %d ‰.\n    Surcoût : %.1f %% pour une clé courte, %.1f %% en moyenne sur %d chiffrements."
                  % (p["a"] + 1, e["seuil_marche"], p["n_a"], p["b"] + 1, e["seuil"], cout_court, cout_moy, p["n_vie"])) + h[i1:]
    h = h.replace("firmware v12, arbre unique", "firmware v13, arbre unique + escalier calibré")
    h = h.replace("Classes : 0 plancher,", "Classes : 0 escalier,")
    i0 = h.index("#define POLICY_MIX_A"); i1 = h.index("#define POLICY_STRONGEST")
    h = h[:i0] + ("#define POLICY_STEP_A         %d          /* marche 1 : niveau %d */\n"
                  "#define POLICY_STEP_B         %d          /* marche 2 : niveau %d */\n"
                  "#define POLICY_STEP_X1        %du        /* x1 <= seuil (‰) -> marche 1 */\n"
                  % (p["a"], p["a"] + 1, p["b"], p["b"] + 1, e["seuil_marche"])) + h[i1:]
    i0 = h.index("/* Arbre unique (v12). x[1]"); i1 = h.index("#endif /* NOISE_POLICY_H_ */")
    corps = ["/* Arbre unique (v13). x[1] = budget consommé (‰). Décision entièrement déterministe. */",
             "static uint8_t policy_decide(const uint16_t *x, const fault_feat_t *f, uint32_t leak_q16, uint32_t limit_q16,",
             "                             uint8_t *cls)",
             "{",
             "    (void)leak_q16;",
             "    if (limit_q16 == 0u) { *cls = 3; return POLICY_STRONGEST; }        /* budget désactivé (banc)          */",
             "    if (policy_fault_alarm(f)) { *cls = 2; return POLICY_STRONGEST; }   /* 1-2. faute : niveau 3 + redondance */",
             "    if (x[1] > %du) { *cls = 1; return POLICY_STRONGEST; }   /* 3. budget épuisé (nœud distillé du planificateur) */"
             % e["seuil"],
             "    *cls = 0;                                                          /* 4. escalier calibré (distillé)   */",
             "    return (x[1] <= POLICY_STEP_X1) ? POLICY_STEP_A : POLICY_STEP_B;",
             "}", "", ""]
    h = h[:i0] + "\n".join(corps) + h[i1:]
    with open(chemin, "w") as fh:
        fh.write(h)
    return dict(chemin=chemin, header=h, cfg=dict(a=p["a"], b=p["b"], seuil_marche=e["seuil_marche"]))


def verifier_c_v13(AU, export, J, n_vie=None):
    """Compile policy_decide v13 sur le PC (gcc) et compare (niveau, classe) C et Python :
    (1) tous les états de faute observés x une grille de budget autour des seuils ;
    (2) sans faute, tous les x1 de 0 à 1300 ;
    (3) rejeu complet du budget de la clé (arithmétique de la carte) sur n_vie + 200 chiffrements."""
    import subprocess, tempfile
    d = tempfile.mkdtemp()
    open(os.path.join(d, "noise_generator.h"), "w").write(
        "#include <stdint.h>\n#define NOISE_TYPE_COUNT 5\nstatic uint8_t noise_rand8(void){return 0;}\n")
    open(os.path.join(d, "noise_policy.h"), "w").write(export["header"])
    T, S = AU["escalier"]["seuil_marche"], AU["budget"]["seuil"]
    U = np.unique(J["X"], axis=0)
    rows = []
    for b in sorted({0, 1, T - 1, T, T + 1, T + 2, 500, S - 1, S, S + 1, S + 2, 1300}):
        lv, c = decide_v13_py(AU, np.full(len(U), b), U)
        rows += [list(u) + [b, l, cc] for u, l, cc in zip(U, lv, c)]
    z = np.array([[0, 0, 0, 255, 0, 0, 0]])      # état « aucune faute » (compteurs initiaux de la carte, gap = 255)
    for b in range(0, 1301):
        lv, c = decide_v13_py(AU, np.array([b]), z)
        rows.append(list(z[0]) + [b, int(lv[0]), int(c[0])])
    np.savetxt(os.path.join(d, "U.txt"), np.array(rows), fmt="%d")
    p = AU["plan"]; n = (n_vie or p["n_vie"]) + 200
    lv_py, _ = simule_budget_carte(lambda x1: int(decide_v13_py(AU, np.array([x1]), z)[0][0]), p["fuites"], p["limite"], n)
    src = r'''#include <stdio.h>
#include "noise_policy.h"
int main(void){ int a[10], bad = 0, n = 0; FILE *u = fopen("%s/U.txt", "r");
  while (fscanf(u, "%%d %%d %%d %%d %%d %%d %%d %%d %%d %%d", a,a+1,a+2,a+3,a+4,a+5,a+6,a+7,a+8,a+9) == 10) {
    fault_feat_t f = { a[0],a[1],a[2],a[3],a[4],a[5],a[6] }; uint16_t x[7] = {0}; uint8_t cls, lv;
    x[1] = (uint16_t)a[7]; lv = policy_decide(x, &f, 0, 1, &cls); bad += (cls != a[9]) || (lv != a[8]); n++; }
  printf("%%d %%d\n", n - bad, n);
  { fault_feat_t f = {0,0,0,255,0,0,0}; uint32_t L = 0; uint8_t cls; int i;
    for (i = 0; i < %d; i++) { uint16_t x[7] = {0}; uint32_t pm = (uint32_t)(((uint64_t)L * 1000u) / POLICY_LEAK_LIMIT_Q16);
      x[1] = pm > 0xFFFFu ? 0xFFFFu : (uint16_t)pm; uint8_t lv = policy_decide(x, &f, L, POLICY_LEAK_LIMIT_Q16, &cls);
      printf("%%d\n", lv); uint64_t s = (uint64_t)L + POLICY_LEVELS[lv].leak_q16; L = s > 0xFFFFFFFFu ? 0xFFFFFFFFu : (uint32_t)s; } }
  return 0; }''' % (d, n)
    open(os.path.join(d, "t.c"), "w").write(src)
    subprocess.run(["gcc", "-O2", "-Wall", "-I" + d, os.path.join(d, "t.c"), "-o", os.path.join(d, "t")], check=True,
                   capture_output=True)
    out = subprocess.run([os.path.join(d, "t")], capture_output=True, text=True).stdout.split("\n")
    ok, tot = map(int, out[0].split())
    lv_c = np.array([int(v) for v in out[1:] if v.strip()])
    return dict(etats_ok=ok, etats=tot, rejeu_ok=int((lv_c == lv_py).sum()), rejeu=n, niveaux_c=lv_c)


def conformite_v13(plogs, AU):
    """Rejoue l'arbre v13 (Python) sur les compteurs lus par la carte (journal 'P') : niveau ET classe doivent
    coïncider sur 100 % des décisions (plus de tirage aléatoire)."""
    bad, ex = 0, []
    for pl in plogs:
        d = lire_plog_v12(pl)
        lv, c = decide_v13_py(AU, np.array([d["x1"]]), np.array([d["feats"]]))
        ok = d["level"] == int(lv[0]) and d["classe"] == int(c[0]) and d["alerte"] == int(c[0] == 2)
        if not ok:
            bad += 1
            if len(ex) < 5:
                ex.append(dict(carte=d, python=(int(lv[0]), int(c[0]))))
    return dict(n=len(plogs), erreurs=bad, conforme=bad == 0, exemples=ex)


def cout_selon_duree(niv, plan, durees=(50, 100, 300, 476, 1000, 2000, 3000, 5000, 10000)):
    """Surcoût moyen d'une clé utilisée n fois : mélange constant v12 contre escalier v13 (mêmes niveaux)."""
    c = [v["surcout"] for v in niv]; a, b = plan["a"], plan["b"]
    p = plan["base"]["p16"] / 65536.0
    rows = []
    for n in durees:
        na = min(n, plan["n_a"])
        rows.append(dict(chiffrements=n, melange_v12=p * c[a] + (1 - p) * c[b],
                         escalier_v13=(na * c[a] + (n - na) * c[b]) / n))
    return pd.DataFrame(rows)
