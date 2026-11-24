"""
politique_v10.py — niveaux v10.1 et plancher calibré, export de noise_policy.h pour le firmware mcu6.

Niveaux = famille du niveau 3 validée (mélange SubBytes / MixColumns / AddRoundKey + délai d avant AddRoundKey,
rondes 0, 1, 10), d = 100 / 200 / 536. N de chaque niveau = N à 10 % de réussite de l'attaquant (plus petit n où
10 % des essais laissent <= 32 bits), minimum sur les campagnes ; non cassé -> borne (n max mesuré).

Budget de fuite (sans changement de clé) : chaque chiffrement au niveau i ajoute 1/N_i ; garantie si la somme
reste <= 1/m sur N_vie chiffrements. Unités du firmware : leak_i = round(2^30 / N_i), limite = round(2^30 / m).
Plancher calibré : mélange aléatoire constant de deux niveaux voisins, de part p pour le moins cher, tel que
p/N_a + (1-p)/N_b = 1/(m N_vie).

Règles embarquées (v10.1, avant l'arbre appris de l'étape 4) :
  x6 > 0 (faute détectée)      -> niveau 3     (classe 2)
  budget épuisé (x1 >= 1000 ‰) -> niveau 3     (classe 1)
  limite = 0 (budget coupé)    -> niveau 3     (classe 3)
  sinon                        -> plancher : niveau a avec la probabilité p, sinon niveau b (classe 0)
"""
import json
import os

import numpy as np

TRACE_ROOT = "/media/khadim/2A53DBCE3700DFDA/trace/v10"
HEADER = "/home/khadim/chipwhisperer/firmware/mcu6/simpleserial-aes/noise_policy.h"
UNIT = 1 << 30
FAM = ("permutation@subbytes_pre:0+permutation@mixcolumns:0+jitter@addroundkey_pre:%d+"
       "permutation@addroundkey_pre:0|m403")
DELAIS = (100, 200, 536)                 # niveaux 1, 2, 3
COMPS = ((1, 3), (3, 3), (4, 1), (4, 3))  # (zone, générateur) : mélange SB, MC, ARK + délai avant ARK
MASK = 0x0403
CLASSES = ("plancher", "budget épuisé", "faute détectée", "budget désactivé")


def n10(r):
    """Plus petit n où 10 % des essais laissent <= 32 bits ; None si jamais atteint."""
    return next((int(n) for n, e in zip(r["ns"], r["effort_p10"]) if e <= 32), None)


def n_prudent10(rs):
    vals = [n10(r) for r in rs]
    cass = [v for v in vals if v is not None]
    return (min(cass), False) if cass else (max(int(r["ns"][-1]) for r in rs), True)


def charger_niveaux(trace_root=TRACE_ROOT):
    ref0 = json.load(open(os.path.join(trace_root, "durees_masque_v10.json")))["000"]
    rv = json.load(open(os.path.join(trace_root, "validation", "resultats_validation.json")))
    rg = json.load(open(os.path.join(trace_root, "validation", "grande", "candidat_50k_resultat_w300.json")))
    niv = []
    for i, d in enumerate(DELAIS):
        rs = [rv[s][FAM % d] for s in ("s0", "s1")] + ([rg] if d == 536 else [])
        N, borne = n_prudent10(rs)
        cyc = float(np.mean([r["trig_mean"] for r in rs]))
        niv.append(dict(niveau=i + 1, d=d, nom=FAM % d, N=N, borne=borne, cycles=cyc,
                        surcout=100 * (cyc / ref0 - 1), leak=int(round(UNIT / N))))
    return niv, ref0


def plancher(niv, n_vie, m):
    """(a, b, p) : niveau a (indice 0..2) avec la probabilité p, sinon b. None si aucun niveau ne suffit."""
    cible = 1.0 / (m * n_vie)
    y = [1.0 / v["N"] for v in niv]
    if cible >= y[0]:
        return 0, 0, 1.0
    for a in range(len(niv) - 1):
        if y[a] >= cible >= y[a + 1]:
            return a, a + 1, (cible - y[a + 1]) / (y[a] - y[a + 1])
    return None


def export_header(niv, n_vie=10000, m=2, path=HEADER):
    pl = plancher(niv, n_vie, m)
    if pl is None:
        raise ValueError("aucun niveau mesuré ne tient N_vie = %d avec m = %d" % (n_vie, m))
    a, b, p = pl
    p16 = min(65535, max(0, int(round(p * 65536))))
    limit = int(round(UNIT / m))
    cout = p * niv[a]["surcout"] + (1 - p) * niv[b]["surcout"]
    lines = []
    for v in niv:
        forces = ", ".join("{%d,%d}" % c for c in COMPS)
        lines.append("    /* niveau %d : %s | surcoût %.1f %% | N(10 %%) %s%d -> fuite 2^30/N */\n"
                     "    { %d, { %s }, { 0, %d, 0, 0, 0 }, 0x%04Xu, %du },"
                     % (v["niveau"], v["nom"], v["surcout"], ">= " if v["borne"] else "", v["N"],
                        len(COMPS), forces, v["d"], MASK, v["leak"]))
    h = """/*
    noise_policy.h — FICHIER GÉNÉRÉ par politique_v10.export_header()  (firmware v10.1)
    Niveaux : famille validée (A4 + 50 000 traces), N à 10 %% de réussite de l'attaquant.
    Budget sans changement de clé : somme de 2^30/N_i <= limite = 2^30/m, m = %d, durée de vie visée N_vie = %d.
    Plancher calibré : niveau %d avec la probabilité %.4f (p16 = %d), sinon niveau %d -> surcoût moyen %.1f %%.
    Règles : faute détectée (x6 > 0) ou budget épuisé -> niveau %d.  Classes : 0 plancher, 1 budget épuisé,
    2 faute détectée, 3 budget désactivé.
*/
#ifndef NOISE_POLICY_H_
#define NOISE_POLICY_H_

#include <stdint.h>
#include "noise_generator.h"

#define POLICY_IS_PLACEHOLDER 0
#define POLICY_N_LEVELS       3
#define POLICY_MAX_FORCES     4
#define POLICY_LEAK_LIMIT_Q16 %du
#define POLICY_MIX_A          %d
#define POLICY_MIX_B          %d
#define POLICY_MIX_P16        %du
#define POLICY_STRONGEST      %d

typedef struct { uint8_t zone; uint8_t gen; } policy_force_t;

typedef struct {
    uint8_t        n_forces;
    policy_force_t forces[POLICY_MAX_FORCES];
    uint16_t       raw[NOISE_TYPE_COUNT];
    uint16_t       round_mask;
    uint32_t       leak_q16;          /* fuite par chiffrement, unité 2^30/N */
} policy_level_t;

static const policy_level_t POLICY_LEVELS[POLICY_N_LEVELS] = {
%s
};

static const uint8_t POLICY_ORDER[POLICY_N_LEVELS] = { 0, 1, 2 };   /* coût croissant */

static uint8_t policy_decide(const uint16_t *x, uint32_t leak_q16, uint32_t limit_q16, uint8_t *cls)
{
    (void)leak_q16;
    if (limit_q16 == 0u) { *cls = 3; return POLICY_STRONGEST; }   /* budget désactivé (banc)           */
    if (x[6] > 0u)       { *cls = 2; return POLICY_STRONGEST; }   /* faute détectée : niveau maximal   */
    if (x[1] >= 1000u)   { *cls = 1; return POLICY_STRONGEST; }   /* budget épuisé : niveau maximal    */
    *cls = 0;                                                     /* plancher calibré (mélange)        */
    {
        uint16_t r = (uint16_t)(((uint16_t)noise_rand8() << 8) | noise_rand8());
        return (r < POLICY_MIX_P16) ? POLICY_MIX_A : POLICY_MIX_B;
    }
}

#endif /* NOISE_POLICY_H_ */
""" % (m, n_vie, a + 1, p, p16, b + 1, cout, len(niv), limit, a, b, p16, len(niv) - 1, "\n".join(lines))
    if path:
        open(path, "w").write(h)
    return dict(a=a, b=b, p=p, p16=p16, limit=limit, cout=cout, header=h)


def decide_py(x, limit=None, r16=None, cfg=None):
    """Réplique Python des règles déterministes. Renvoie (niveau, classe) ; pour le plancher (classe 0), le
    niveau dépend du tirage r16 : si r16 est None, renvoie (None, 0) = 'a ou b'."""
    if limit == 0:
        return 2, 3
    if x[6] > 0:
        return 2, 2
    if x[1] >= 1000:
        return 2, 1
    if r16 is None:
        return None, 0
    return (cfg["a"] if r16 < cfg["p16"] else cfg["b"]), 0


def lire_plog(plog):
    """Journal 'P' v10.1 (16 octets) -> dict(level, x[0..6], decide_cycles, epuise, classe)."""
    b = [int(v) for v in plog]
    x = [(b[1 + 2 * i] << 8) | b[2 + 2 * i] for i in range(6)] + [(b[15] >> 1) & 0x07]
    return dict(level=b[0], x=x, decide_cycles=(b[13] << 8) | b[14], epuise=b[15] & 1, classe=(b[15] >> 4) & 0x07)


def conformite(plogs, cfg, alpha_z=4.0):
    """Règles déterministes : 100 % attendu. Plancher : niveau dans {a, b} et part de a compatible avec p."""
    bad, n_pl, n_a, det = 0, 0, 0, 0
    for pl in plogs:
        d = lire_plog(pl)
        lv, cl = decide_py(d["x"], limit=cfg.get("limit"))
        if cl != 0:
            det += 1
            bad += int(d["level"] != lv or d["classe"] != cl)
        else:
            n_pl += 1
            bad += int(d["level"] not in (cfg["a"], cfg["b"]) or d["classe"] != 0)
            n_a += int(d["level"] == cfg["a"])
    p, z = cfg["p"], None
    if n_pl:
        sd = np.sqrt(p * (1 - p) / n_pl) if 0 < p < 1 else 0
        z = (n_a / n_pl - p) / sd if sd > 0 else 0.0
    return dict(n=len(plogs), erreurs=bad, deterministes=det, plancher=n_pl,
                part_a=(n_a / n_pl if n_pl else None), p_attendu=p, z=z,
                conforme=(bad == 0 and (z is None or abs(z) < alpha_z)))


if __name__ == "__main__":
    niv, ref0 = charger_niveaux()
    for v in niv:
        print("niveau %d (d = %3d) : %6.0f cycles  %+5.1f %%  N(10 %%) %s%d  fuite %d"
              % (v["niveau"], v["d"], v["cycles"], v["surcout"], ">= " if v["borne"] else "", v["N"], v["leak"]))
    out = export_header(niv)
    print("plancher : niveau %d avec p = %.4f, sinon niveau %d -> surcoût moyen %.1f %% | limite %d"
          % (out["a"] + 1, out["p"], out["b"] + 1, out["cout"], out["limit"]))
    print("écrit :", HEADER)
