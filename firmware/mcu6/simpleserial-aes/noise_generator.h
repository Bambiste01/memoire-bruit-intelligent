/*
    noise_generator.h — v10 « bruit conçu par optimisation + politique embarquée + détection de fautes »

    v10 (corrections du générateur, chemin SCA modifié -> nouvelle référence sans bruit obligatoire) :
         * PRNG xoshiro128** ensemencé par l'ADC (VREFINT + capteur de température), repli sur graine fixe ;
         * délais à moyenne flottante (Coron & Kizhvatov, CHES 2010), réglables par la commande 'J' ;
         * chemin rapide des délais dans protect_zone() ;
         * aes.c : créneaux factices de SubBytes et de MixColumns sur le MÊME chemin de code que les réels ;
         * commande 'E' : version du firmware, état de la graine, réglage de la moyenne flottante ;
         * compilation « production » (EXTRA_OPTS=NOISE_PRODUCTION) sans les commandes de banc.
         Générateurs NON utilisés en v10 (laissés pour compatibilité, hors de l'espace de ParEGO) :
         NOISE_DUMMY_BALANCED (écriture du complément APRÈS la valeur : ce n'est pas un double rail) et
         NOISE_ALU (n'agit que comme un délai).

    v9 : surveillance des fautes (FIA), SANS changer le chemin SCA (désactivée par défaut) :
         * calcul redondant optionnel (mode 0 = off, 1 = toujours, 2 = auto) : le chiffrement est exécuté deux
           fois ; si les deux résultats diffèrent, la sortie est SUPPRIMÉE (octets aléatoires) et la détection
           est comptée (cf. contre-mesures par redondance temporelle) ;
         * suivi des redémarrages : cause du dernier reset (RCC_CSR) + compteur conservé en RAM (.noinit) ;
         * ligne 'H' (8 octets) : indicateurs de santé (détections récentes, redémarrages récents, cause) ;
         * cible de faute dédiée (commande 'F') : contrôle d'autorisation (branche conditionnelle) précédé des
           mêmes zones de bruit que l'AES, pour mesurer le taux de CONTOURNEMENT d'un glitch synchronisé sur le
           trigger, avec / sans bruit / avec détection.

    v8 : le générateur NOISE_PERMUTATION devient un vrai mélange (shuffling) des opérations de la ronde :
         SubBytes fusionné avec ShiftRows (16 créneaux + jusqu'à 16 factices), MixColumns (4 colonnes + jusqu'à
         4 factices), AddRoundKey (16 créneaux). Intensité brute de la permutation = nombre de colonnes factices
         de MixColumns (et 4x ce nombre d'octets factices dans SubBytes). Les délais aléatoires des zones
         SUBBYTES_BYTE / MIXCOLUMNS / ADDROUNDKEY_BYTE s'insèrent ENTRE les créneaux mélangés.

    Changements par rapport à v6 :
      * plus aucune feature endogène (le firmware n'observe plus son propre bruit) ;
      * politique adaptative (noise_policy.h, généré par policy_train.py) = classifieur de session appris
        (arbre ENTIER sur l'état exogène du moniteur) + planificateur à budget de fuite ;
      * décision prise AVANT le chiffrement : le journal 'P' de l'AES i décrit exactement la config de l'AES i ;
      * budget de fuite par clé : L += atanh(rho)²(niveau) à chaque AES (Mangard 2004 : N = 3 + 2z²/atanh(rho)²) ;
      * masque de rondes : le bruit forcé ne s'applique qu'aux rondes sélectionnées (réduit le surcoût) ;
      * horodatage des zones (commande 'T') : correspondance exacte zone -> échantillon pour la carte de fuite.
*/
#ifndef NOISE_GENERATOR_H_
#define NOISE_GENERATOR_H_

#include <stdint.h>

#define NOISE_FW_VERSION 13   /* renvoyé par la commande 'E'                           */
#define FW_INFO_LEN         4   /* ligne 'e' : version, graine ADC, changements du LSB, fm_shift */
#define MONITOR_RECORD_LEN 19   /* ligne 'V' (inchangée depuis v6)                       */
#define POLICY_N_FEATURES   7   /* entrées de l'arbre embarqué (v10.1 : + x6 indices de faute) */
#define POLICY_X6_MAX       7   /* x6 saturé : stocké sur 3 bits dans le journal 'P'         */
#define POLICY_LOG_LEN     16   /* ligne 'P' (journal de décision, une par AES adaptatif) */
#define ZONE_TS_LEN        28   /* ligne 'T' : 7 zones x u32                              */
#define HEALTH_LEN          8   /* ligne 'H' (v9)                                         */
#define FAULT_SIG_LEN       8   /* ligne 'G' (v10.1) : signaux de faute du dernier chiffrement */
#define FAULT_GRANTED  0x600D0000u   /* résultat du contrôle d'autorisation : accès accordé  */
#define FAULT_DENIED   0xDEAD0000u   /* accès refusé (résultat normal)                       */

typedef enum {
    NOISE_NONE = 0,
    NOISE_RANDOM_DELAY,
    NOISE_DUMMY_BALANCED,
    NOISE_PERMUTATION,
    NOISE_ALU,
    NOISE_TYPE_COUNT
} noise_type_t;

typedef enum {
    INTENSITY_LOW = 0,
    INTENSITY_MEDIUM,
    INTENSITY_HIGH,
    INTENSITY_COUNT
} noise_intensity_t;

typedef struct {
    uint8_t activer_mask;
    noise_intensity_t intensity[NOISE_TYPE_COUNT];
} decision_vec_t;

typedef enum {
    ZONE_ROUND_START = 0,
    ZONE_SUBBYTES_PRE,
    ZONE_SUBBYTES_BYTE,
    ZONE_MIXCOLUMNS,
    ZONE_ADDROUNDKEY_PRE,
    ZONE_ADDROUNDKEY_BYTE,
    ZONE_ROUND_END,
    ZONE_COUNT
} injection_zone_t;

#define NOISE_MAX_ROUNDS 11u            /* rondes 0 (AddRoundKey initial) à 10 */
#define ROUND_MASK_ALL   0x07FFu

/* Initialisation */
void noise_generator_init(void);
void noise_on_plaintext(const uint8_t *pt);

/* v10 : graine et délais à moyenne flottante */
uint8_t noise_seed_status(void);        /* 1 = graine tirée de l'ADC, 0 = graine de repli */
uint8_t noise_seed_lsb_changes(void);   /* changements du bit de poids faible de l'ADC pendant l'ensemencement */
void    noise_set_fm_shift(uint8_t s);  /* b = a >> s ; 0 = délais uniformes indépendants (v9) ; défaut 2 */
uint8_t noise_get_fm_shift(void);

/* Configuration (mode forcé, utilisé par l'optimiseur côté hôte) */
void noise_set_mode(uint8_t mode_classique);
void noise_set_fixed_intensity(noise_intensity_t intensity);
void noise_set_logging(uint8_t enabled);
void noise_set_round_mask(uint16_t mask);       /* bit r = 1 : bruit actif à la ronde r */
uint16_t noise_get_round_mask(void);

/* Compatibilité v6 : valeurs mémorisées mais SANS effet (l'heuristique risk-map est retirée) */
void noise_set_risk(uint8_t round, injection_zone_t zone, uint8_t risk_raw);
void noise_set_generator_weight(noise_type_t generator, uint8_t weight_raw);
void noise_set_budget_cycles(uint16_t budget);

/* Forçage */
void noise_force_generator(injection_zone_t zone, noise_type_t generator, noise_intensity_t intensity);
void noise_clear_override(injection_zone_t zone);
void noise_clear_all_overrides(void);

/* Surveillance */
void noise_begin_encryption(void);
void noise_end_encryption(void);
void noise_dump_summary(void);
void noise_set_per_event_log(uint8_t enabled);

/* Points d'injection (appelés par aes.c) */
void noise_zone_round_start(uint8_t round);
void noise_zone_subbytes_pre(uint8_t order[4], uint8_t round);
void noise_zone_subbytes_byte(volatile uint8_t *byte_ptr, uint8_t round);
void noise_zone_mixcolumns(uint8_t round);
void noise_zone_addroundkey_pre(uint8_t round);
void noise_zone_addroundkey_byte(volatile uint8_t *byte_ptr, uint8_t round);
void noise_zone_round_end(uint8_t round);

/* Mode adaptatif (politique apprise) */
void    noise_enable_adaptive(uint8_t enabled);
uint8_t noise_is_adaptive(void);
void    noise_decide_and_apply(void);           /* appeler AVANT le chiffrement */
void    noise_set_leak_limit(uint32_t limit_q16);
uint8_t noise_policy_fill_log(uint8_t *buf);    /* écrit POLICY_LOG_LEN octets */

/* Debug / observation */
void noise_get_current_config(uint8_t *zone, uint8_t *gen, uint8_t *intensity);
void noise_get_observation(uint8_t *obs, uint8_t len);   /* = features courantes, format 'P' */

/* Mélange v8 (appelé par aes.c) : remplit order[0..n-1] avec une permutation aléatoire si le générateur
   NOISE_PERMUTATION est actif pour (zone, ronde), l'identité sinon. Les indices >= n_real sont des créneaux
   FACTICES ; leur nombre vaut min(intensité brute de la permutation x dummy_scale, max_dummy).
   order doit pouvoir contenir n_real + max_dummy éléments. Retourne n. */
uint8_t noise_get_order(injection_zone_t zone, uint8_t round, uint8_t *order,
                        uint8_t n_real, uint8_t dummy_scale, uint8_t max_dummy);
uint8_t noise_rand8(void);
uint8_t noise_last_order_active(void);   /* v10 : 1 si le dernier noise_get_order() a mélangé (sinon ordre identité, sans factice) */

/* ========== Détection de fautes (v9) ==========
   Ligne 'G' (v10.1, FAULT_SIG_LEN = 8 octets, big-endian), émise avec 'H' quand le rapport est actif :
     [0..2] net_cycles   durée du DERNIER chiffrement moins les cycles de délais insérés (u24 saturé)
     [3..5] total_cycles durée du dernier chiffrement (u24 saturé)
     [6]    round_err_recent  erreurs du compteur de rondes récentes (décroissance /2 toutes les 256 op.)
     [7]    flags  bit0 = erreur de ronde pendant CETTE opération, bit1 = désaccord de redondance,
                   bit2 = calcul redondant actif
   Ligne 'H' (HEALTH_LEN = 8 octets, big-endian) :
     [0..1] detect_recent   u16  détections récentes (divisé par 2 tous les 256 chiffrements)
     [2]    boots_recent    u8   redémarrages récents (idem)
     [3]    flags           bit0 = dernière vérification en désaccord ; bit1 = redondance active pour le dernier
                            chiffrement ; bit2 = démarrage à froid (mémoire conservée invalide) ;
                            bits 4-7 = cause du dernier reset (0 inconnue, 1 broche, 2 mise sous tension,
                            3 logiciel, 4 chien de garde indépendant, 5 chien de garde fenêtré, 6 basse
                            consommation, 7 option bytes)
     [4..5] detect_total    u16  détections depuis la mise sous tension
     [6..7] redundant_runs  u16  chiffrements exécutés en double */
void     fault_set_mode(uint8_t mode);          /* 0 off, 1 toujours, 2 auto (règle simple, voir .c) */
uint8_t  fault_get_mode(void);
void     fault_set_report(uint8_t enabled);     /* émission de la ligne 'H' après chaque opération */
uint8_t  fault_report_enabled(void);
uint8_t  fault_redundancy_active(void);         /* décision pour l'opération qui commence */
uint8_t  fault_compare(const uint8_t *a, const uint8_t *b, uint8_t n);   /* 1 = désaccord (enregistré) */
void     fault_on_operation_done(void);         /* décroissance périodique des compteurs récents */
void     fault_fill_health(uint8_t *buf);       /* HEALTH_LEN octets */
void     fault_clear_counters(void);
void     noise_fault_round_error(void);         /* v10.1 : compteur de rondes incohérent (appelé par aes.c) */
void     fault_op_begin(void);                  /* v10.1 : début d'opération (remet le drapeau de ronde)     */
uint8_t  fault_round_error_this_op(void);       /* v10.1 : 1 si une ronde manquante a été détectée          */
void     fault_fill_signals(uint8_t *buf);      /* v10.1 : FAULT_SIG_LEN octets (ligne 'G')                 */
uint32_t fault_loop_once(void);                 /* cible n°2 : boucle de comptage, GRANTED si cnt != 1600 */
uint32_t fault_check_once(void);               /* contrôle d'autorisation : FAULT_DENIED en fonctionnement normal */
void     noise_fault_gate(void);                /* zones de bruit placées avant l'opération sensible */

/* Horodatage des zones (carte de fuite) */
void noise_arm_zone_timestamps(void);           /* enregistre les zones du PROCHAIN chiffrement */
void noise_get_zone_timestamps(uint8_t round, uint8_t *buf);   /* ZONE_TS_LEN octets, big-endian */

/* Intensité brute */
void     noise_set_raw_intensity(noise_type_t gen, uint16_t cycles);
void     noise_clear_raw_intensity(void);
uint16_t noise_get_raw_intensity(noise_type_t gen);
uint8_t  noise_is_raw_active(noise_type_t gen);

/* ========== Moniteur SCA (features exogènes, aucune dépendance au bruit ni à la clé) ==========
   Unité de temps : 1 tick = 1024 cycles CPU.
   Enregistrement 'V' (MONITOR_RECORD_LEN = 19 octets, big-endian) — inchangé depuis v6 :
     [0..1]   n_since_key   u16 (saturé)
     [2..5]   dt_last       u32 ticks
     [6..9]   dt_ema        u32 ticks
     [10..13] dt_dev        u32 ticks
     [14]     hd_last       u8
     [15..16] hd_ema_q4     u16
     [17]     pt_repeat     u8
     [18]     key_changes   u8 (saturé)

   Journal de décision 'P' (POLICY_LOG_LEN = 16 octets, big-endian) :
     [0]      level         niveau choisi pour CE chiffrement (0xFF = hors mode adaptatif)
     [1..2]   x0 n_since_key        (u16 saturé)
     [3..4]   x1 leak_permille      budget de fuite consommé AVANT ce chiffrement, ‰ de la limite (u16 saturé)
     v12 : [5] rs16  [6] rs64  [7] boots  [8] gap  [9] det  [10] rnd  [11] dur  [12] 0
           (compteurs de faute lus par l'arbre unique ; x2..x5 restent disponibles dans la ligne 'V')
     [13..14] decide_cycles         coût de la décision (cycles, u16 saturé)
     [15]     flags                 bit0 = budget épuisé ; bits 1-3 = x6 (v12 : alerte de la branche faute, 0/1) ;
                                    bits 4-6 = classe / motif de décision
   x6 (v10.1) = min(7, fautes détectées récentes + redémarrages récents au-delà du premier).
   Les 7 features sont stockées EXACTEMENT comme la politique les lit : rejouer la politique Python sur ce
   journal doit redonner `level` et la classe dans 100 % des cas (policy_train.conformity_report).
*/
void    monitor_reset(void);
void    monitor_on_key_loaded(const uint8_t *key);   /* v10.1 : même clé -> budget conservé */
void    noise_budget_forget(void);                   /* v10.1 : banc seulement ('R')          */
void    monitor_on_plaintext(const uint8_t *pt);
uint8_t monitor_fill_record(uint8_t *buf);

#endif /* NOISE_GENERATOR_H_ */