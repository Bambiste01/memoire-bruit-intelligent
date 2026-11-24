/*
    noise_policy.h — FICHIER GÉNÉRÉ par detecteur_v10.export_politique()  (firmware v13, arbre unique + escalier calibré)
    Niveaux : famille validée (A4 + 50 000 traces), N à 10 % de réussite de l'attaquant.
    Budget sans changement de clé : somme de 2^30/N_i <= limite = 2^30/m, m = 2, durée de vie visée N_vie = 10000.
    Escalier calibré (v13) : niveau 2 tant que x1 <= 236 ‰ (475 premiers chiffrements), puis niveau 3 ; budget épuisé au-delà de 999 ‰.
    Surcoût : 28.3 % pour une clé courte, 42.6 % en moyenne sur 10000 chiffrements.
    Arbre unique : sûreté (faute prouvée) / resets appris / budget distillé -> niveau 3.  Classes : 0 escalier, 1 budget épuisé,
    2 faute détectée, 3 budget désactivé.
    Entraîné sur 6 campagnes de glitch + 4 sessions normales (cellules 10.2, 10.2b, 11.1).
*/
#ifndef NOISE_POLICY_H_
#define NOISE_POLICY_H_

#include <stdint.h>
#include "noise_generator.h"

#define POLICY_IS_PLACEHOLDER 0
#define POLICY_N_LEVELS       3
#define POLICY_MAX_FORCES     4
#define POLICY_LEAK_LIMIT_Q16 536870912u
#define POLICY_STEP_A         1          /* marche 1 : niveau 2 */
#define POLICY_STEP_B         2          /* marche 2 : niveau 3 */
#define POLICY_STEP_X1        236u        /* x1 <= seuil (‰) -> marche 1 */
#define POLICY_STRONGEST      2

typedef struct { uint8_t zone; uint8_t gen; } policy_force_t;

typedef struct {
    uint8_t        n_forces;
    policy_force_t forces[POLICY_MAX_FORCES];
    uint16_t       raw[NOISE_TYPE_COUNT];
    uint16_t       round_mask;
    uint32_t       leak_q16;          /* fuite par chiffrement, unité 2^30/N */
} policy_level_t;

static const policy_level_t POLICY_LEVELS[POLICY_N_LEVELS] = {
    /* niveau 1 : permutation@subbytes_pre:0+permutation@mixcolumns:0+jitter@addroundkey_pre:100+permutation@addroundkey_pre:0|m403 | surcoût 23.8 % | N(10 %) 1200 -> fuite 2^30/N */
    { 4, { {1,3}, {3,3}, {4,1}, {4,3} }, { 0, 100, 0, 0, 0 }, 0x0403u, 894785u },
    /* niveau 2 : permutation@subbytes_pre:0+permutation@mixcolumns:0+jitter@addroundkey_pre:200+permutation@addroundkey_pre:0|m403 | surcoût 28.3 % | N(10 %) 4000 -> fuite 2^30/N */
    { 4, { {1,3}, {3,3}, {4,1}, {4,3} }, { 0, 200, 0, 0, 0 }, 0x0403u, 268435u },
    /* niveau 3 : permutation@subbytes_pre:0+permutation@mixcolumns:0+jitter@addroundkey_pre:536+permutation@addroundkey_pre:0|m403 | surcoût 43.3 % | N(10 %) >= 25000 -> fuite 2^30/N */
    { 4, { {1,3}, {3,3}, {4,1}, {4,3} }, { 0, 536, 0, 0, 0 }, 0x0403u, 42950u },
};

static const uint8_t POLICY_ORDER[POLICY_N_LEVELS] = { 0, 1, 2 };   /* coût croissant */

typedef struct { uint8_t rs16, rs64, boots, gap, det, rnd, dur; } fault_feat_t;   /* compteurs de faute */

/* Arbre unique (v12), branche « faute » : nœuds 1 et 2. Utilisée par policy_decide ET par la redondance
   automatique (mode 2), pour que la redondance suive exactement la classe 2 de l'arbre. Fonction pure. */
static inline uint8_t policy_fault_alarm(const fault_feat_t *f)
{
    if (f->det | f->rnd | f->dur) return 1;   /* 1. sûreté : faute prouvée */
    /* 2. sous-arbre appris sur les resets (monotone, profondeur 3, kFA 1) */
    if (f->boots <= 3) {
        if (f->gap <= 42) {
            if (f->gap <= 13) {
                return 1;   /* resets suspects (appris) */
            } else {
                return 0;   /* resets compatibles avec un usage normal */
            }
        } else {
            if (f->gap <= 114) {
                return 0;   /* resets compatibles avec un usage normal */
            } else {
                return 0;   /* resets compatibles avec un usage normal */
            }
        }
    } else {
        if (f->rs64 <= 1) {
            return 1;   /* resets suspects (appris) */
        } else {
            return 1;   /* resets suspects (appris) */
        }
    }
}

/* Arbre unique (v13). x[1] = budget consommé (‰). Décision entièrement déterministe. */
static uint8_t policy_decide(const uint16_t *x, const fault_feat_t *f, uint32_t leak_q16, uint32_t limit_q16,
                             uint8_t *cls)
{
    (void)leak_q16;
    if (limit_q16 == 0u) { *cls = 3; return POLICY_STRONGEST; }        /* budget désactivé (banc)          */
    if (policy_fault_alarm(f)) { *cls = 2; return POLICY_STRONGEST; }   /* 1-2. faute : niveau 3 + redondance */
    if (x[1] > 999u) { *cls = 1; return POLICY_STRONGEST; }   /* 3. budget épuisé (nœud distillé du planificateur) */
    *cls = 0;                                                          /* 4. escalier calibré (distillé)   */
    return (x[1] <= POLICY_STEP_X1) ? POLICY_STEP_A : POLICY_STEP_B;
}

#endif /* NOISE_POLICY_H_ */
