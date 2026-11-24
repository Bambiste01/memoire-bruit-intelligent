/*
    simpleserial-aes.c — v10 (v9 + corrections du générateur de bruit, voir noise_generator.h)

    Nouvelles commandes v10 :
      'E' 1 octet  : renvoie 'e' (4 octets) = version du firmware (10), graine ADC (1/0),
                     changements du bit de poids faible de l'ADC pendant l'ensemencement, fm_shift
      'J' 1 octet  : délais à moyenne flottante, b = a >> s (défaut 2) ; 0 = délais uniformes de v9
    Compilation « production » : make PLATFORM=CWLITEARM CRYPTO_TARGET=TINYAES128C SS_VER=SS_VER_1_1 \
                                 EXTRA_OPTS=NOISE_PRODUCTION
      -> seules 'k', 'p', 'x', 'E', 'H' restent ; la politique embarquée est toujours active.

    Nouvelles commandes v9 :
      'W' 2 octets : [mode de redondance 0/1/2][1 = émettre la ligne 'H' après chaque opération]
      'H' 1 octet  : 0 = lit la ligne 'H' ; 1 = remet les compteurs de fautes à zéro puis lit
      'F' 1 octet  : (réservé) cible de faute : contrôle d'autorisation, renvoie 'f' (5 octets :
                     résultat u32 big-endian + drapeaux : bit0 = désaccord détecté, bit1 = redondance active)

    Ordre d'un chiffrement 'p' :
      1. noise_on_plaintext()      : moniteur exogène (exposition, cadence, Hamming, répétition)
      2. noise_decide_and_apply()  : (mode adaptatif) l'arbre choisit le niveau de CE chiffrement,
                                     AVANT le trigger -> la décision n'entre pas dans la trace
      3. trigger_high / AES / trigger_low
      4. lignes émises : 'r' (chiffré), 'V' (moniteur), 'P' (journal de décision, mode adaptatif seulement)

    Nouvelles commandes v7 :
      'M' 2 octets : masque de rondes (big-endian), bit r = bruit actif à la ronde r (0x0002 = ronde 1 seule)
      'B' 4 octets : limite du budget de fuite (Q16, big-endian) ; 0 = budget désactivé
      'T' 1 octet  : 0xFF = arme l'horodatage du prochain chiffrement ; r = renvoie la ligne 'T' de la ronde r
    Commandes conservées : k p x o i l z g b c d a Q O I V R
    ('z', 'g', 'b' sont acceptées mais sans effet depuis v7.)
*/
#include "aes-independant.h"
#include "hal.h"
#include "simpleserial.h"
#include "noise_generator.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint8_t set_adaptive(uint8_t* c, uint8_t len) { noise_enable_adaptive(c[0]); return 0x00; }
uint8_t get_mask(uint8_t* m, uint8_t len)     { aes_indep_mask(m, len); return 0x00; }
uint8_t get_key(uint8_t* k, uint8_t len)      { aes_indep_key(k); monitor_on_key_loaded(k); return 0x00; }

static void emit_health_if_enabled(void)
{
    if (fault_report_enabled()) {
        uint8_t h[HEALTH_LEN];
        fault_fill_health(h);
        simpleserial_put('H', HEALTH_LEN, h);
        uint8_t g[FAULT_SIG_LEN];                    /* v10.1 : signaux de faute du dernier chiffrement */
        fault_fill_signals(g);
        simpleserial_put('G', FAULT_SIG_LEN, g);
    }
}

uint8_t get_pt(uint8_t* pt, uint8_t len)
{
    noise_on_plaintext(pt);
    fault_op_begin();                                /* v10.1 */
    if (noise_is_adaptive()) noise_decide_and_apply();

    if (!fault_redundancy_active()) {
        /* chemin d'origine (SCA) : strictement identique à v8 */
        aes_indep_enc_pretrigger(pt);
        trigger_high();
      #ifdef ADD_JITTER
        for (volatile uint8_t k = 0; k < (*pt & 0x0F); k++);
      #endif
        aes_indep_enc(pt);
        trigger_low();
        aes_indep_enc_posttrigger(pt);
    } else {
        /* calcul redondant : deux exécutions (chacune avec son propre mélange / ses propres délais) ;
           en cas de désaccord, la sortie fautive n'est JAMAIS livrée */
        uint8_t b[16];
        memcpy(b, pt, 16);
        aes_indep_enc_pretrigger(pt);
        aes_indep_enc_pretrigger(b);
        trigger_high();
        aes_indep_enc(pt);
        aes_indep_enc(b);
        trigger_low();
        aes_indep_enc_posttrigger(pt);
        aes_indep_enc_posttrigger(b);
        if (fault_compare(pt, b, 16)) {
            for (uint8_t i = 0; i < 16; i++) pt[i] = noise_rand8();
        }
    }
    if (fault_round_error_this_op())                 /* v10.1 : ronde manquante -> sortie jamais livrée */
        for (uint8_t i = 0; i < 16; i++) pt[i] = noise_rand8();
    fault_on_operation_done();

    simpleserial_put('r', 16, pt);

    {
        uint8_t rec[MONITOR_RECORD_LEN];
        monitor_fill_record(rec);
        simpleserial_put('V', MONITOR_RECORD_LEN, rec);
    }
    if (noise_is_adaptive()) {
        uint8_t log[POLICY_LOG_LEN];
        noise_policy_fill_log(log);
        simpleserial_put('P', POLICY_LOG_LEN, log);
    }
    emit_health_if_enabled();
    return 0x00;
}

/* --- v9 : détection de fautes --- */
uint8_t set_fault_mode(uint8_t* c, uint8_t len) { fault_set_mode(c[0]); fault_set_report(c[1]); return 0x00; }

uint8_t get_health(uint8_t* c, uint8_t len)
{
    if (c[0] == 1) fault_clear_counters();
    uint8_t h[HEALTH_LEN];
    fault_fill_health(h);
    simpleserial_put('H', HEALTH_LEN, h);
    return 0x00;
}

/* Cible de faute : le trigger est levé AVANT le bruit, comme pour l'AES, puis le contrôle est exécuté. Un glitch
   synchronisé sur le trigger (ext_offset fixe) doit donc viser un instant que les délais rendent variable. */
uint8_t fault_bench(uint8_t* c, uint8_t len)
{
    static uint32_t counter = 0;
    uint8_t fake[16] = {0};
    fake[0] = (uint8_t)(counter >> 24); fake[1] = (uint8_t)(counter >> 16);
    fake[2] = (uint8_t)(counter >> 8);  fake[3] = (uint8_t)counter;
    counter++;
    noise_on_plaintext(fake);
    if (noise_is_adaptive()) noise_decide_and_apply();

    uint8_t red = fault_redundancy_active();
    uint32_t r1, r2 = 0;
    trigger_high();
    noise_fault_gate();
    uint8_t loop_target = (len >= 1 && c[0] == 1);   /* 0 : comparaison ; 1 : boucle de comptage */
    r1 = loop_target ? fault_loop_once() : fault_check_once();
    if (red) {
        noise_fault_gate();
        r2 = loop_target ? fault_loop_once() : fault_check_once();
    }
    trigger_low();

    uint8_t mismatch = 0;
    if (red) {
        uint8_t a[4] = {(uint8_t)(r1 >> 24), (uint8_t)(r1 >> 16), (uint8_t)(r1 >> 8), (uint8_t)r1};
        uint8_t b[4] = {(uint8_t)(r2 >> 24), (uint8_t)(r2 >> 16), (uint8_t)(r2 >> 8), (uint8_t)r2};
        mismatch = fault_compare(a, b, 4);
        if (mismatch) r1 = FAULT_DENIED;               /* sortie supprimée : on ne livre pas un accès accordé */
    }
    fault_on_operation_done();

    uint8_t out[5] = {(uint8_t)(r1 >> 24), (uint8_t)(r1 >> 16), (uint8_t)(r1 >> 8), (uint8_t)r1,
                      (uint8_t)((mismatch ? 0x01 : 0x00) | (red ? 0x02 : 0x00))};
    simpleserial_put('f', 5, out);
    emit_health_if_enabled();
    return 0x00;
}

uint8_t get_external_features(uint8_t* c, uint8_t len)
{
    uint8_t rec[MONITOR_RECORD_LEN];
    monitor_fill_record(rec);
    simpleserial_put('V', MONITOR_RECORD_LEN, rec);
    return 0x00;
}

uint8_t monitor_reset_cmd(uint8_t* c, uint8_t len) { monitor_reset(); noise_budget_forget(); return 0x00; }   /* banc */
uint8_t reset(uint8_t* x, uint8_t len) { return 0x00; }

/* --- Contrôle du bruit --- */
uint8_t set_noise_mode(uint8_t* c, uint8_t len)      { noise_set_mode(c[0]); return 0x00; }
uint8_t set_noise_intensity(uint8_t* c, uint8_t len) { noise_set_fixed_intensity((noise_intensity_t)c[0]); return 0x00; }
uint8_t set_noise_logging(uint8_t* c, uint8_t len)   { noise_set_logging(c[0]); return 0x00; }
uint8_t set_noise_risk(uint8_t* c, uint8_t len)      { noise_set_risk(c[0], (injection_zone_t)c[1], c[2]); return 0x00; }
uint8_t set_noise_gen_weight(uint8_t* c, uint8_t len){ noise_set_generator_weight((noise_type_t)c[0], c[1]); return 0x00; }
uint8_t set_noise_budget(uint8_t* c, uint8_t len)    { noise_set_budget_cycles(((uint16_t)c[0] << 8) | c[1]); return 0x00; }
uint8_t set_noise_force(uint8_t* c, uint8_t len)     { noise_force_generator((injection_zone_t)c[0], (noise_type_t)c[1], (noise_intensity_t)c[2]); return 0x00; }

uint8_t set_noise_unforce(uint8_t* c, uint8_t len)
{
    if (c[0] == 0xFF) noise_clear_all_overrides();
    else              noise_clear_override((injection_zone_t)c[0]);
    return 0x00;
}

uint8_t get_current_config(uint8_t* c, uint8_t len)
{
    uint8_t cfg[3];
    noise_get_current_config(&cfg[0], &cfg[1], &cfg[2]);
    simpleserial_put('Q', 3, cfg);
    return 0x00;
}

uint8_t get_observation(uint8_t* c, uint8_t len)
{
    uint8_t obs[POLICY_LOG_LEN];
    noise_get_observation(obs, POLICY_LOG_LEN);
    simpleserial_put('O', POLICY_LOG_LEN, obs);
    return 0x00;
}

uint8_t set_noise_intensity_raw(uint8_t* c, uint8_t len)
{
    if (c[0] == 0xFF) noise_clear_raw_intensity();
    else              noise_set_raw_intensity((noise_type_t)c[0], ((uint16_t)c[1] << 8) | c[2]);
    return 0x00;
}

/* --- Nouvelles commandes v7 --- */
uint8_t set_round_mask(uint8_t* c, uint8_t len) { noise_set_round_mask(((uint16_t)c[0] << 8) | c[1]); return 0x00; }

uint8_t set_leak_limit(uint8_t* c, uint8_t len)
{
    uint32_t v = ((uint32_t)c[0] << 24) | ((uint32_t)c[1] << 16) | ((uint32_t)c[2] << 8) | c[3];
    noise_set_leak_limit(v);
    return 0x00;
}

/* --- v10 --- */
/* 'E' : renvoie 'e' = [version][1 = graine ADC][changements du LSB de l'ADC][fm_shift]. La graine elle-même
   n'est jamais renvoyée. */
uint8_t fw_info(uint8_t* c, uint8_t len)
{
    uint8_t b[FW_INFO_LEN] = {NOISE_FW_VERSION, noise_seed_status(), noise_seed_lsb_changes(), noise_get_fm_shift()};
    simpleserial_put('e', FW_INFO_LEN, b);
    return 0x00;
}
/* 'J' : réglage des délais à moyenne flottante (b = a >> s ; 0 = délais uniformes de v9). */
uint8_t set_fm_shift(uint8_t* c, uint8_t len) { noise_set_fm_shift(c[0]); return 0x00; }

uint8_t zone_timestamps(uint8_t* c, uint8_t len)
{
    if (c[0] == 0xFF) { noise_arm_zone_timestamps(); return 0x00; }
    uint8_t buf[ZONE_TS_LEN];
    noise_get_zone_timestamps(c[0], buf);
    simpleserial_put('T', ZONE_TS_LEN, buf);
    return 0x00;
}

#if SS_VER == SS_VER_2_1
uint8_t aes(uint8_t cmd, uint8_t scmd, uint8_t len, uint8_t *buf) {
    uint8_t req_len = 0, err = 0, mask_len = 0;
    if (scmd & 0x04) { mask_len = buf[req_len]; req_len += 1 + mask_len; if (req_len > len) return SS_ERR_LEN; err = get_mask(buf + req_len - mask_len, mask_len); if (err) return err; }
    if (scmd & 0x02) { req_len += 16; if (req_len > len) return SS_ERR_LEN; err = get_key(buf + req_len - 16, 16); if (err) return err; }
    if (scmd & 0x01) { req_len += 16; if (req_len > len) return SS_ERR_LEN; err = get_pt(buf + req_len - 16, 16); if (err) return err; }
    if (len != req_len) return SS_ERR_LEN;
    return 0x00;
}
#endif

int main(void)
{
    uint8_t tmp[KEY_LENGTH] = {DEFAULT_KEY};
    platform_init(); init_uart(); trigger_setup();
    putch('A'); aes_indep_init(); putch('B');
    aes_indep_key(tmp); putch('C');
    simpleserial_init(); putch('S');

    #if SS_VER == SS_VER_2_1
    simpleserial_addcmd(0x01, 16, aes);
    #else
    simpleserial_addcmd('k', 16, get_key);
    simpleserial_addcmd('p', 16, get_pt);
    simpleserial_addcmd('x',  0, reset);
    simpleserial_addcmd('E',  1, fw_info);
    #ifndef NOISE_PRODUCTION
    /* commandes de banc : absentes de la version « production » (make ... EXTRA_OPTS=NOISE_PRODUCTION),
       où seule la politique embarquée choisit le bruit */
    simpleserial_addcmd('J',  1, set_fm_shift);
    simpleserial_addcmd('o',  1, set_noise_mode);
    simpleserial_addcmd('i',  1, set_noise_intensity);
    simpleserial_addcmd('l',  1, set_noise_logging);
    simpleserial_addcmd('z',  3, set_noise_risk);
    simpleserial_addcmd('g',  2, set_noise_gen_weight);
    simpleserial_addcmd('b',  2, set_noise_budget);
    simpleserial_addcmd('c',  3, set_noise_force);
    simpleserial_addcmd('d',  1, set_noise_unforce);
    simpleserial_addcmd('a',  1, set_adaptive);
    simpleserial_addcmd('Q',  1, get_current_config);
    simpleserial_addcmd('O',  1, get_observation);
    simpleserial_addcmd('I',  3, set_noise_intensity_raw);
    simpleserial_addcmd('V',  1, get_external_features);
    simpleserial_addcmd('R',  1, monitor_reset_cmd);
    simpleserial_addcmd('M',  2, set_round_mask);
    simpleserial_addcmd('B',  4, set_leak_limit);
    simpleserial_addcmd('T',  1, zone_timestamps);
    simpleserial_addcmd('W',  2, set_fault_mode);
    simpleserial_addcmd('H',  1, get_health);
    simpleserial_addcmd('F',  1, fault_bench);
    #else
    noise_enable_adaptive(1);                    /* production : politique embarquée toujours active */
    fault_set_mode(2);                           /* production : redondance automatique sur détection   */
    simpleserial_addcmd('H',  1, get_health);
    #endif
    #endif

    while (1) simpleserial_get();
}