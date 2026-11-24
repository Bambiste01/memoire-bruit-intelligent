/*
    noise_policy.h — FICHIER GÉNÉRÉ par politique_v10.export_header()  (firmware v10.1)
    Niveaux : famille validée (A4 + 50 000 traces), N à 10 % de réussite de l'attaquant.
    Budget sans changement de clé : somme de 2^30/N_i <= limite = 2^30/m, m = 2, durée de vie visée N_vie = 10000.
    Plancher calibré : niveau 2 avec la probabilité 0.0476 (p16 = 3121), sinon niveau 3 -> surcoût moyen 42.6 %.
    Règles : faute détectée (x6 > 0) ou budget épuisé -> niveau 3.  Classes : 0 plancher, 1 budget épuisé,
    2 faute détectée, 3 budget désactivé.
*/
#ifndef NOISE_POLICY_H_
#define NOISE_POLICY_H_

#include <stdint.h>
#include "noise_generator.h"

#define POLICY_IS_PLACEHOLDER 0
#define POLICY_N_LEVELS       3
#define POLICY_MAX_FORCES     4
#define POLICY_LEAK_LIMIT_Q16 536870912u
#define POLICY_MIX_A          1
#define POLICY_MIX_B          2
#define POLICY_MIX_P16        3121u
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
