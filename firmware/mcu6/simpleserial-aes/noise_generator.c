/*
    noise_generator.c — v10 (v9 + corrections du générateur, voir noise_generator.h)

    v10 :
      * PRNG xoshiro128** (Blackman & Vigna) au lieu de xorshift32, ensemencé au démarrage par le bruit des bits
        de poids faible de l'ADC (VREFINT et capteur de température), toujours re-mélangé par le compteur de
        cycles à chaque plaintext. Repli sur l'ancienne graine si l'ADC ne répond pas (indiqué par la commande 'E').
      * Délais à moyenne flottante (Coron & Kizhvatov, CHES 2010) : pour chaque chiffrement, une moyenne m est
        tirée dans [0, a - b] ; chaque délai vaut m + r, r uniforme dans [0, b), avec a = intensité brute et
        b = a >> g_fm_shift. Coût moyen identique à v9 (a/2 itérations). g_fm_shift = 0 redonne exactement les
        délais uniformes indépendants de v9 (commande 'J').
      * Chemin rapide : une zone qui n'a que des délais (intensité brute, sans journalisation) exécute le délai
        directement, sans construire de décision ni mesurer le coût -> surcoût des points d'injection réduit.

    Architecture (voir noise_generator.h) :
      OBSERVER  : moniteur exogène (exposition de la clé, cadence, Hamming, répétition des plaintexts)
      PRÉDIRE   : classifieur de session appris (arbre entier, noise_policy.h) -> horizon d'exposition H
      ADAPTER   : planificateur à budget de fuite : L = somme des atanh(rho)² du niveau utilisé depuis le
                  chargement de la clé ; niveau le moins coûteux tel que L + fuite x (H - n) <= limite
      OPTIMISER : les niveaux eux-mêmes viennent de l'optimisation multi-objectif hors ligne (noise_opt.py)

    Le firmware n'utilise plus aucune feature dérivée de son propre bruit (boucle circulaire de v4-v6).
*/
#include "noise_generator.h"
#include "noise_policy.h"
#include <stddef.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
/* Registres DWT (Cortex-M4)                                               */
/* ---------------------------------------------------------------------- */
#if defined(__arm__)
#define DEMCR      (*(volatile uint32_t*)0xE000EDFCu)
#define DWT_CTRL   (*(volatile uint32_t*)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t*)0xE0001004u)
#else                                   /* test fonctionnel sur PC (gcc) : registres simulés */
static volatile uint32_t host_demcr, host_dwt_ctrl, host_cyccnt;
#define DEMCR      host_demcr
#define DWT_CTRL   host_dwt_ctrl
#define DWT_CYCCNT host_cyccnt
#endif

extern void putch(char c);

/* ---------------------------------------------------------------------- */
/* État des générateurs                                                    */
/* ---------------------------------------------------------------------- */
static uint8_t  g_mode_classique = 0;
static noise_intensity_t g_fixed_intensity = INTENSITY_MEDIUM;
static uint8_t  g_logging_enabled = 0;
static uint8_t  g_per_event_log = 0;
static uint16_t g_round_mask = ROUND_MASK_ALL;

/* Compatibilité v6 (mémorisés, sans effet) */
static uint8_t  g_risk_map[NOISE_MAX_ROUNDS][ZONE_COUNT];
static uint8_t  g_generator_weight[NOISE_TYPE_COUNT];
static uint16_t g_budget_cycles_per_zone = 600;

/* v8 : NOISE_PERMUTATION = mélange de l'ordre des opérations (noise_get_order), sur SubBytes+ShiftRows
   (16 octets), MixColumns (4 colonnes + colonnes factices) et AddRoundKey (16 octets). */
static const uint8_t zone_eligible_mask[ZONE_COUNT] = {
    (1u << NOISE_RANDOM_DELAY),                                                           /* ROUND_START     */
    (1u << NOISE_PERMUTATION) | (1u << NOISE_RANDOM_DELAY),                               /* SUBBYTES_PRE    */
    (1u << NOISE_DUMMY_BALANCED) | (1u << NOISE_RANDOM_DELAY),                            /* SUBBYTES_BYTE   */
    (1u << NOISE_ALU) | (1u << NOISE_RANDOM_DELAY) | (1u << NOISE_PERMUTATION),           /* MIXCOLUMNS      */
    (1u << NOISE_RANDOM_DELAY) | (1u << NOISE_PERMUTATION),                               /* ADDROUNDKEY_PRE */
    (1u << NOISE_DUMMY_BALANCED) | (1u << NOISE_RANDOM_DELAY),   /* ADDROUNDKEY_BYTE*/
    (1u << NOISE_RANDOM_DELAY),                                  /* ROUND_END       */
};

static uint8_t           g_override_mask[ZONE_COUNT];
static noise_intensity_t g_override_intensity[ZONE_COUNT][NOISE_TYPE_COUNT];
static uint16_t          g_raw_cycles[NOISE_TYPE_COUNT];
static uint8_t           g_use_raw[NOISE_TYPE_COUNT];

/* Compteurs de coût (journalisation 'S' uniquement — jamais entrées de la politique) */
static uint16_t g_events_this_enc = 0;
static uint16_t g_events_per_zone[ZONE_COUNT];
static uint32_t g_total_event_cycles = 0;
static uint32_t g_enc_start_cycle = 0;
static uint32_t g_last_enc_cycles = 0;                  /* v10.1 : durée du dernier chiffrement   */
static uint32_t g_last_net_cycles = 0;                  /* v10.1 : même durée moins les délais      */

/* Horodatage des zones */
static uint8_t  g_ts_state = 0;                         /* 0 inactif, 1 armé, 2 enregistrement */
static uint32_t g_zone_ts[NOISE_MAX_ROUNDS][ZONE_COUNT];

/* Politique adaptative */
static uint8_t  g_adaptive_enabled = 0;
static uint8_t  g_cur_level = 0xFF;                      /* niveau du chiffrement en cours      */
static uint8_t  g_applied_level = 0xFF;                  /* niveau actuellement chargé          */
static uint16_t g_x[POLICY_N_FEATURES];                  /* features de la dernière décision    */
static uint16_t g_decide_cycles = 0;
static uint8_t  g_cur_class = 0;                         /* classe de session prédite            */
static uint32_t g_leak_q16 = 0;                          /* budget de fuite consommé (par clé)  */
static uint32_t g_leak_limit_q16 = POLICY_LEAK_LIMIT_Q16;

/* v10.1 : budget de fuite conservé lors d'un reset (zone .noinit), lié à l'empreinte de la clé. Sans cela,
   un reset (ou un nouvel envoi de la MÊME clé) remettait le budget à zéro. Limite : une coupure d'alimentation
   efface la RAM (une sauvegarde en flash par paquets serait nécessaire : perspective). */
#define BUD_MAGIC 0xB0D6E7A1u
typedef struct { uint32_t magic; uint32_t key_fp; uint32_t leak; uint32_t chk; } budget_nv_t;
static budget_nv_t g_bud __attribute__((section(".noinit")));
static uint32_t bud_chk(const budget_nv_t *b) { return b->magic ^ b->key_fp ^ (b->leak * 0x9E3779B1u) ^ 0x5A5AA5A5u; }
static void bud_save(void) { if (g_bud.magic == BUD_MAGIC) { g_bud.leak = g_leak_q16; g_bud.chk = bud_chk(&g_bud); } }
static uint32_t key_fingerprint(const uint8_t *k)
{
    uint32_t h = 0x811C9DC5u;                                   /* FNV-1a 32 bits (empreinte, pas un secret) */
    for (uint8_t i = 0; i < 16; i++) { h ^= k[i]; h *= 0x01000193u; }
    return h;
}

/* ---------------------------------------------------------------------- */
/* Moniteur exogène                                                        */
/* ---------------------------------------------------------------------- */
#define MON_TICK_SHIFT 10
#define MON_TICK_MAX   0x000FFFFFu
#define MON_HIST       8

static uint32_t mon_n_since_key;
static uint32_t mon_dt_last;
static uint32_t mon_dt_ema;             /* ticks, Q8 */
static uint32_t mon_dt_dev;             /* ticks, Q8 */
static uint32_t mon_hd_ema;             /* Q8        */
static uint8_t  mon_hd_last;
static uint8_t  mon_pt_repeat;
static uint8_t  mon_key_changes;
static uint32_t mon_last_cyccnt;
static uint8_t  mon_have_last_time;
static uint8_t  mon_dt_primed;
static uint8_t  mon_hd_primed;
static uint8_t  mon_last_pt[16];
static uint8_t  mon_have_last_pt;
static uint8_t  mon_pt_history[MON_HIST][16];
static uint8_t  mon_pt_hist_n;
static uint8_t  mon_pt_hist_idx;

/* ---------------------------------------------------------------------- */
/* Utilitaires                                                             */
/* ---------------------------------------------------------------------- */
static uint16_t sat16(uint32_t v) { return (v > 0xFFFFu) ? 0xFFFFu : (uint16_t)v; }

static uint32_t sat_add32(uint32_t a, uint32_t b)
{
    uint32_t s = a + b;
    return (s < a) ? 0xFFFFFFFFu : s;
}

static void put_be16(uint8_t *b, uint16_t v) { b[0] = (uint8_t)(v >> 8); b[1] = (uint8_t)v; }
static void put_be32(uint8_t *b, uint32_t v)
{
    b[0] = (uint8_t)(v >> 24); b[1] = (uint8_t)(v >> 16);
    b[2] = (uint8_t)(v >> 8);  b[3] = (uint8_t)v;
}

static void putc_hex(uint8_t v)
{
    static const char hexd[] = "0123456789ABCDEF";
    putch(hexd[(v >> 4) & 0xF]);
    putch(hexd[v & 0xF]);
}
static void putc_hex16(uint16_t v) { putc_hex((uint8_t)(v >> 8)); putc_hex((uint8_t)v); }
static void putc_hex24(uint32_t v) { putc_hex((uint8_t)(v >> 16)); putc_hex((uint8_t)(v >> 8)); putc_hex((uint8_t)v); }

/* ---------------------------------------------------------------------- */
/* PRNG v10 : xoshiro128** ensemencé par l'ADC, re-mélangé à chaque plaintext */
/* ---------------------------------------------------------------------- */
static uint32_t g_rng_s[4];
static uint8_t  g_seed_status = 0;      /* 1 = graine ADC, 0 = repli (ADC indisponible)             */
static uint8_t  g_seed_lsb_changes = 0; /* nombre de changements du bit de poids faible (sat. 255)  */

static inline uint32_t rotl32(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static uint32_t rng_next(void)
{
    uint32_t *s = g_rng_s;
    uint32_t r = rotl32(s[1] * 5u, 7) * 9u;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11);
    return r;
}

/* Finaliseur de murmur3 : diffuse chaque bit d'entrée sur tout le mot (mélange de la graine). */
static uint32_t mix32(uint32_t h)
{
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

/* ADC1 du STM32F303 en accès registre (RM0316). Horloge synchrone HCLK/1 (CKMODE = 01, valable car le
   prescaler AHB vaut 1 sur la carte ChipWhisperer). Toutes les attentes ont un délai maximal : en cas
   d'échec, la fonction rend 0 et le générateur garde la graine de repli. */
#if defined(__arm__)
#define RCC_AHBENR   (*(volatile uint32_t*)0x40021014u)
#define ADC1_ISR     (*(volatile uint32_t*)0x50000000u)
#define ADC1_CR      (*(volatile uint32_t*)0x50000008u)
#define ADC1_CFGR    (*(volatile uint32_t*)0x5000000Cu)
#define ADC1_SMPR2   (*(volatile uint32_t*)0x50000018u)
#define ADC1_SQR1    (*(volatile uint32_t*)0x50000030u)
#define ADC1_DR      (*(volatile uint32_t*)0x50000040u)
#define ADC12_CCR    (*(volatile uint32_t*)0x50000308u)
#define ADC_TIMEOUT  200000u

static int adc_wait(volatile uint32_t *reg, uint32_t mask, uint32_t want)
{
    for (uint32_t k = 0; k < ADC_TIMEOUT; k++)
        if (((*reg) & mask) == want) return 1;
    return 0;
}

static uint8_t adc_seed(uint32_t out[4])
{
    RCC_AHBENR |= (1u << 28);                                   /* ADC12EN */
    ADC12_CCR = (ADC12_CCR & ~(3u << 16)) | (1u << 16)          /* CKMODE = HCLK/1 */
              | (1u << 22) | (1u << 23);                        /* VREFEN, TSEN    */
    ADC1_CR = (ADC1_CR & ~(3u << 28));                          /* ADVREGEN : 10 -> 00 */
    ADC1_CR = (ADC1_CR & ~(3u << 28)) | (1u << 28);             /*            00 -> 01 (activé) */
    for (volatile uint32_t k = 0; k < 2000u; k++) __asm volatile ("nop");   /* > 10 µs */
    ADC1_CR &= ~(1u << 30);                                     /* ADCALDIF = 0 (entrées simples) */
    ADC1_CR |= (1u << 31);                                      /* ADCAL */
    if (!adc_wait(&ADC1_CR, (1u << 31), 0)) return 0;
    for (volatile uint32_t k = 0; k < 100u; k++) __asm volatile ("nop");
    ADC1_ISR = (1u << 0);                                       /* efface ADRDY */
    ADC1_CR |= (1u << 0);                                       /* ADEN */
    if (!adc_wait(&ADC1_ISR, (1u << 0), (1u << 0))) return 0;
    ADC1_CFGR = 0;                                              /* 12 bits, conversion unique, logiciel */
    ADC1_SMPR2 &= ~((7u << 18) | (7u << 24));                   /* temps d'échantillonnage minimal (bruit max) */

    uint32_t acc[4] = {0x243F6A88u, 0x85A308D3u, 0x13198A2Eu, 0x03707344u};
    uint8_t  last = 0, changes = 0;
    for (uint32_t k = 0; k < 512u; k++) {
        uint32_t ch = (k & 1u) ? 16u : 18u;                     /* capteur de température / VREFINT */
        ADC1_SQR1 = (ch << 6);                                  /* L = 0 (1 conversion), SQ1 = ch */
        ADC1_CR |= (1u << 2);                                   /* ADSTART */
        if (!adc_wait(&ADC1_ISR, (1u << 2), (1u << 2))) return 0;
        uint32_t v = ADC1_DR;                                   /* lit et efface EOC */
        uint8_t lsb = (uint8_t)(v & 1u);
        if (k > 1 && lsb != last && changes < 255) changes++;
        last = lsb;
        acc[k & 3u] = mix32(acc[k & 3u] ^ (v << ((k >> 2) & 15u)) ^ DWT_CYCCNT);
    }
    ADC1_CR |= (1u << 1);                                       /* ADDIS : ADC arrêté pendant les AES */
    (void)adc_wait(&ADC1_CR, (1u << 0), 0);
    for (int i = 0; i < 4; i++) out[i] = acc[i];
    g_seed_lsb_changes = changes;
    return 1;
}
#else
static uint8_t adc_seed(uint32_t out[4]) { (void)out; return 0; }   /* test sur hôte */
#endif

static void rng_seed(void)
{
    uint32_t s[4] = {0xA5A5A5A5u, 0x3C6EF372u, 0xBB67AE85u, 0x510E527Fu};   /* repli : graine fixe (v9) */
    g_seed_status = adc_seed(s);
    for (int i = 0; i < 4; i++) g_rng_s[i] = mix32(s[i] + (uint32_t)i * 0x9E3779B9u);
    if ((g_rng_s[0] | g_rng_s[1] | g_rng_s[2] | g_rng_s[3]) == 0) g_rng_s[0] = 0xDEADBEEFu;
    for (int i = 0; i < 16; i++) (void)rng_next();
}

void noise_on_plaintext(const uint8_t *pt)
{
    g_rng_s[0] ^= mix32(DWT_CYCCNT);                            /* instant d'arrivée : entropie supplémentaire */
    if ((g_rng_s[0] | g_rng_s[1] | g_rng_s[2] | g_rng_s[3]) == 0) g_rng_s[0] = 0xDEADBEEFu;
    monitor_on_plaintext(pt);
}

uint8_t noise_seed_status(void)      { return g_seed_status; }
uint8_t noise_seed_lsb_changes(void) { return g_seed_lsb_changes; }

/* ---------------------------------------------------------------------- */
/* Délais à moyenne flottante (v10)                                          */
/* ---------------------------------------------------------------------- */
static uint8_t  g_fm_shift = 2;          /* b = a >> g_fm_shift ; 0 = délais uniformes indépendants (v9) */
static uint32_t g_fm_draw = 0;           /* tirage du chiffrement en cours (fixe la moyenne m)          */

static inline void delay_loop(uint32_t n)
{
    for (volatile uint32_t i = 0; i < n; i++) __asm volatile ("nop");
}

static inline uint32_t fm_length(uint32_t a)
{
    if (a == 0) return 0;
    if (g_fm_shift == 0) return rng_next() % a;                 /* comportement v9 */
    uint32_t b = a >> g_fm_shift;
    if (b == 0) b = 1;
    uint32_t span = (a > b) ? (a - b) : 0;                      /* m dans [0, a - b] */
    uint32_t m = span ? (g_fm_draw % (span + 1u)) : 0;
    return m + (rng_next() % b);
}

void    noise_set_fm_shift(uint8_t s) { g_fm_shift = (s > 8) ? 8 : s; }
uint8_t noise_get_fm_shift(void)      { return g_fm_shift; }

/* ---------------------------------------------------------------------- */
/* Générateurs (inchangés depuis v6)                                       */
/* ---------------------------------------------------------------------- */
static uint32_t gen_random_delay(noise_intensity_t intensity, uint32_t raw, void *ctx)
{
    (void)ctx;
    static const uint16_t preset[INTENSITY_COUNT] = {50, 150, 400};
    uint32_t max_c = raw > 0 ? raw : preset[intensity];
    uint32_t n = fm_length(max_c);                              /* v10 : moyenne flottante */
    delay_loop(n);
    return n;
}

static uint32_t gen_dummy_balanced(noise_intensity_t intensity, uint32_t raw, void *ctx)
{
    static const uint8_t preset[INTENSITY_COUNT] = {1, 3, 6};
    volatile uint8_t *data = (volatile uint8_t *)ctx;
    uint8_t n = raw > 0 ? (raw > 255 ? 255 : (uint8_t)raw) : preset[intensity];
    for (uint8_t i = 0; i < n; i++) {
        volatile uint8_t d = (data != 0) ? (uint8_t)(~(*data)) : (uint8_t)(~rng_next());
        (void)d;
    }
    return n;
}

static uint32_t gen_permutation(noise_intensity_t intensity, uint32_t raw, void *ctx)
{
    static const uint8_t preset[INTENSITY_COUNT] = {1, 2, 3};
    uint8_t *order = (uint8_t *)ctx;
    if (!order) return 0;
    uint8_t n = raw > 0 ? (raw > 255 ? 255 : (uint8_t)raw) : preset[intensity];
    for (uint8_t i = 0; i < n; i++) {
        uint8_t a = rng_next() % 4, b = rng_next() % 4;
        uint8_t t = order[a]; order[a] = order[b]; order[b] = t;
    }
    return n;
}

static uint32_t gen_alu(noise_intensity_t intensity, uint32_t raw, void *ctx)
{
    (void)ctx;
    static const uint8_t preset[INTENSITY_COUNT] = {5, 20, 50};
    volatile uint32_t acc = rng_next();
    uint32_t n = raw > 0 ? raw : preset[intensity];
    for (uint32_t i = 0; i < n; i++) acc = (acc * 2654435761u) ^ (acc >> 3);
    return n;
}

typedef uint32_t (*noise_fn_t)(noise_intensity_t, uint32_t, void *);
static const noise_fn_t noise_table[NOISE_TYPE_COUNT] = {
    0, gen_random_delay, gen_dummy_balanced, gen_permutation, gen_alu
};

/* ---------------------------------------------------------------------- */
/* Moniteur exogène (identique à v6)                                       */
/* ---------------------------------------------------------------------- */
static uint8_t popcnt8(uint8_t v)
{
    v = (uint8_t)(v - ((v >> 1) & 0x55u));
    v = (uint8_t)((v & 0x33u) + ((v >> 2) & 0x33u));
    return (uint8_t)((v + (v >> 4)) & 0x0Fu);
}

void monitor_reset(void)
{
    mon_n_since_key = 0;
    mon_dt_last = 0; mon_dt_ema = 0; mon_dt_dev = 0;
    mon_hd_ema = 0;  mon_hd_last = 0;
    mon_pt_repeat = 0;
    mon_key_changes = 0;
    mon_have_last_time = 0; mon_dt_primed = 0; mon_hd_primed = 0;
    mon_have_last_pt = 0;
    mon_pt_hist_n = 0; mon_pt_hist_idx = 0;
    memset(mon_last_pt, 0, sizeof(mon_last_pt));
    memset(mon_pt_history, 0, sizeof(mon_pt_history));
    g_leak_q16 = 0;
}

void monitor_on_key_loaded(const uint8_t *key)
{
    uint32_t fp = key ? key_fingerprint(key) : 0u;
    mon_n_since_key = 0;
    if (g_bud.magic == BUD_MAGIC && g_bud.chk == bud_chk(&g_bud) && g_bud.key_fp == fp) {
        g_leak_q16 = g_bud.leak;                 /* même clé : le budget déjà consommé est conservé */
    } else {
        g_leak_q16 = 0;                          /* clé différente (ou démarrage à froid) : budget neuf */
        g_bud.magic = BUD_MAGIC; g_bud.key_fp = fp; g_bud.leak = 0; g_bud.chk = bud_chk(&g_bud);
        if (mon_key_changes < 255) mon_key_changes++;
    }
}

void noise_budget_forget(void)                   /* banc seulement ('R') : budget remis à zéro pour la clé */
{                                                /* courante ; la mémorisation reste ARMÉE (v10.1, corrigé) */
    g_leak_q16 = 0;
    if (g_bud.magic == BUD_MAGIC) { g_bud.leak = 0; g_bud.chk = bud_chk(&g_bud); }
}

void monitor_on_plaintext(const uint8_t *pt)
{
    if (!pt) return;
    uint32_t now = DWT_CYCCNT;

    if (mon_n_since_key < 0xFFFFFFFFu) mon_n_since_key++;

    if (!mon_have_last_time) {
        mon_dt_last = 0;
        mon_have_last_time = 1;
    } else {
        uint32_t t = (now - mon_last_cyccnt) >> MON_TICK_SHIFT;
        if (t > MON_TICK_MAX) t = MON_TICK_MAX;
        mon_dt_last = t;
        if (!mon_dt_primed) {
            mon_dt_ema = t << 8;
            mon_dt_dev = 0;
            mon_dt_primed = 1;
        } else {
            uint32_t mean = mon_dt_ema >> 8;
            uint32_t diff = (t > mean) ? (t - mean) : (mean - t);
            mon_dt_ema = (mon_dt_ema * 3u + (t << 8)) >> 2;
            mon_dt_dev = (mon_dt_dev * 3u + (diff << 8)) >> 2;
        }
    }
    mon_last_cyccnt = now;

    if (mon_have_last_pt) {
        uint8_t hd = 0;
        for (uint8_t i = 0; i < 16; i++) hd += popcnt8((uint8_t)(pt[i] ^ mon_last_pt[i]));
        mon_hd_last = hd;
        if (!mon_hd_primed) { mon_hd_ema = (uint32_t)hd << 8; mon_hd_primed = 1; }
        else                { mon_hd_ema = (mon_hd_ema * 3u + ((uint32_t)hd << 8)) >> 2; }
    } else {
        mon_hd_last = 0;
    }

    mon_pt_repeat = 0;
    for (uint8_t k = 0; k < mon_pt_hist_n; k++) {
        if (memcmp(pt, mon_pt_history[k], 16) == 0) { mon_pt_repeat = 1; break; }
    }
    memcpy(mon_pt_history[mon_pt_hist_idx], pt, 16);
    mon_pt_hist_idx = (uint8_t)((mon_pt_hist_idx + 1u) & (MON_HIST - 1u));
    if (mon_pt_hist_n < MON_HIST) mon_pt_hist_n++;

    memcpy(mon_last_pt, pt, 16);
    mon_have_last_pt = 1;
}

uint8_t monitor_fill_record(uint8_t *b)
{
    if (!b) return 0;
    put_be16(b, sat16(mon_n_since_key));
    put_be32(b + 2,  mon_dt_last);
    put_be32(b + 6,  mon_dt_ema >> 8);
    put_be32(b + 10, mon_dt_dev >> 8);
    b[14] = mon_hd_last;
    put_be16(b + 15, sat16(mon_hd_ema >> 4));
    b[17] = mon_pt_repeat;
    b[18] = mon_key_changes;
    return MONITOR_RECORD_LEN;
}

/* ---------------------------------------------------------------------- */
/* Application du bruit dans une zone                                      */
/* ---------------------------------------------------------------------- */
static decision_vec_t zone_decision(injection_zone_t zone)
{
    decision_vec_t d;
    d.activer_mask = 0;
    for (uint8_t g = 0; g < NOISE_TYPE_COUNT; g++) d.intensity[g] = INTENSITY_LOW;

    if (g_override_mask[zone] != 0) {                     /* config forcée (hôte ou politique) */
        d.activer_mask = g_override_mask[zone] & zone_eligible_mask[zone];
        for (uint8_t g = 0; g < NOISE_TYPE_COUNT; g++) d.intensity[g] = g_override_intensity[zone][g];
    } else if (g_mode_classique) {                        /* contre-mesure « tout activé »      */
        uint8_t eligible = zone_eligible_mask[zone];
        for (uint8_t g = 0; g < NOISE_TYPE_COUNT; g++)
            if (eligible & (1u << g)) { d.activer_mask |= (1u << g); d.intensity[g] = g_fixed_intensity; }
    }
    /* sinon : aucun bruit (l'heuristique risk-map de v6 est retirée) */
    return d;
}

static void protect_zone(injection_zone_t zone, uint8_t round, void *ctx)
{
    if (g_ts_state == 2 && round < NOISE_MAX_ROUNDS && g_zone_ts[round][zone] == 0xFFFFFFFFu)
        g_zone_ts[round][zone] = DWT_CYCCNT - g_enc_start_cycle;

    if (round < 16 && !(g_round_mask & (1u << round))) return;

    /* chemin rapide (v8.1) : aucune config pour cette zone -> sortie immédiate, sans construire la décision */
    if (g_override_mask[zone] == 0 && !g_mode_classique) return;

    /* chemin rapide (v10) : la zone n'a que des délais à intensité brute et la journalisation est coupée ->
       délai exécuté directement (pas de décision, pas de mesure de coût). La permutation, gérée par
       noise_get_order(), peut être active en plus sur la même zone. */
    if (!g_mode_classique && !g_logging_enabled && g_use_raw[NOISE_RANDOM_DELAY]) {
        uint8_t m = (uint8_t)(g_override_mask[zone] & zone_eligible_mask[zone] & ~(1u << NOISE_PERMUTATION));
        if (m == (1u << NOISE_RANDOM_DELAY)) {
            uint32_t t0 = DWT_CYCCNT;
            delay_loop(fm_length(g_raw_cycles[NOISE_RANDOM_DELAY]));
            g_total_event_cycles += DWT_CYCCNT - t0;     /* v10.1 : durée nette = durée - délais insérés */
            return;
        }
    }

    decision_vec_t d = zone_decision(zone);
    if (d.activer_mask == 0) return;

    for (uint8_t g = 1; g < NOISE_TYPE_COUNT; g++) {
        if (!(d.activer_mask & (1u << g))) continue;
        if (g == NOISE_PERMUTATION) continue;            /* v8 : géré par noise_get_order() */
        uint32_t raw = g_use_raw[g] ? g_raw_cycles[g] : 0;
        uint32_t t0 = DWT_CYCCNT;
        noise_table[g](d.intensity[g], raw, ctx);
        uint32_t elapsed = DWT_CYCCNT - t0;
        g_events_this_enc++;
        g_events_per_zone[zone]++;
        g_total_event_cycles += elapsed;
        if (g_logging_enabled && g_per_event_log) {
            putch('N'); putch(' '); putc_hex((uint8_t)zone); putch(' '); putc_hex(round); putch(' ');
            putc_hex(g); putch(' '); putc_hex16(sat16(elapsed)); putch('\n');
        }
    }
}


/* ---------------------------------------------------------------------- */
/* Surveillance des fautes (v9)                                            */
/* ---------------------------------------------------------------------- */
/* v12 : TOUS les compteurs de faute sont en zone .noinit (conservés lors d'un reset si l'alimentation tient),
   y compris le compteur d'opérations qui rythme leur décroissance. En v10.1 ce compteur était en RAM volatile :
   il repartait de 0 à chaque reset, donc les redémarrages récents ne décroissaient jamais si les resets étaient
   espacés de moins de 256 opérations (fausses alertes mesurées en 11.1). Logique identique au simulateur
   detecteur_v10.simule(decroissance_persistante=True) sur lequel l'arbre a été entraîné. */
#define NV_MAGIC 0xFA17C012u                 /* v12 : nouveau magic -> démarrage à froid après mise à jour */
typedef struct {
    uint32_t magic;
    uint8_t  boots;      /* resets récents, plafond 4, demi-vie 256 opérations (= boots_recent v10.1)   */
    uint8_t  rs16;       /* resets récents, plafond 15, divisé par 2 toutes les 16 opérations            */
    uint8_t  rs64;       /* resets récents, plafond 15, divisé par 2 toutes les 64 opérations            */
    uint8_t  gap;        /* opérations depuis le dernier reset (saturé à 255)                            */
    uint8_t  det;        /* désaccords de redondance récents, plafond 4, demi-vie 256                    */
    uint8_t  rnd;        /* erreurs du compteur de rondes récentes, plafond 4, demi-vie 256              */
    uint8_t  dur;        /* durées nettes anormales récentes, plafond 4, demi-vie 256                    */
    uint8_t  pad;
    uint16_t ops;        /* compteur d'opérations (rythme des décroissances)                              */
    uint16_t pad2;
} fault_nv_t;
/* Zone .noinit : non initialisée par le démarrage. Un magic invalide = démarrage à froid. */
static fault_nv_t g_nv __attribute__((section(".noinit")));

#define FAULT_CNT_CAP   4u
#define FAULT_RS_CAP    15u
#define FAULT_NET_REF   47520u   /* v13 : 47 520 mesuré (diagnostic 15.3, 400 op.) ; v12 : 47 543 (cellule 9.5)  */
#define FAULT_NET_TOL   25u      /* seuil d'anomalie (normal : écart max 15 cycles sur 2 400 opérations)       */

static uint8_t  g_red_mode = 0;            /* 0 off, 1 toujours, 2 auto (= classe 2 de l'arbre)           */
static uint8_t  g_fault_report = 0;
static uint16_t g_detect_total = 0;
static uint16_t g_red_runs = 0;
static uint8_t  g_last_mismatch = 0;
static uint8_t  g_last_red = 0;
static uint8_t  g_reset_cause = 0;
static uint8_t  g_cold_boot = 1;
static uint16_t g_round_err_total = 0;
static uint8_t  g_round_err_op = 0;        /* erreur de ronde pendant l'opération en cours                  */
static uint8_t  g_op_is_aes = 0;           /* v12 : l'opération en cours est un AES (pas le banc 'F')       */

/* RCC_CSR du STM32F3 (0x40021024) : bits 25 OBL, 26 PIN, 27 POR, 28 SFT, 29 IWDG, 30 WWDG, 31 LPWR ; 24 = RMVF */
#if defined(__arm__)
#define RCC_CSR (*(volatile uint32_t*)0x40021024u)
static uint8_t read_reset_cause(void)
{
    uint32_t c = RCC_CSR;
    RCC_CSR = c | (1u << 24);                       /* efface les indicateurs pour le prochain démarrage */
    if (c & (1u << 29)) return 4;
    if (c & (1u << 30)) return 5;
    if (c & (1u << 31)) return 6;
    if (c & (1u << 28)) return 3;
    if (c & (1u << 27)) return 2;
    if (c & (1u << 26)) return 1;
    if (c & (1u << 25)) return 7;
    return 0;
}
#else
static uint8_t read_reset_cause(void) { return 0; }   /* test sur hôte */
#endif

static void nv_zero(void)
{
    g_nv.boots = 0; g_nv.rs16 = 0; g_nv.rs64 = 0; g_nv.gap = 255;
    g_nv.det = 0; g_nv.rnd = 0; g_nv.dur = 0; g_nv.pad = 0; g_nv.ops = 0; g_nv.pad2 = 0;
}

static void fault_mon_init(void)
{
    g_reset_cause = read_reset_cause();
    if (g_nv.magic == NV_MAGIC) {                   /* reset à chaud : un redémarrage de plus */
        if (g_nv.boots < FAULT_CNT_CAP) g_nv.boots++;
        if (g_nv.rs16 < FAULT_RS_CAP)   g_nv.rs16++;
        if (g_nv.rs64 < FAULT_RS_CAP)   g_nv.rs64++;
        g_nv.gap = 0;
        if (g_nv.boots > FAULT_CNT_CAP) g_nv.boots = FAULT_CNT_CAP;
        if (g_nv.det > FAULT_CNT_CAP)   g_nv.det = FAULT_CNT_CAP;
        if (g_nv.rnd > FAULT_CNT_CAP)   g_nv.rnd = FAULT_CNT_CAP;
        if (g_nv.dur > FAULT_CNT_CAP)   g_nv.dur = FAULT_CNT_CAP;
        g_cold_boot = 0;
    } else {                                        /* démarrage à froid : RAM effacée */
        g_nv.magic = NV_MAGIC;
        nv_zero();
        g_cold_boot = 1;
    }
    g_red_mode = 0; g_fault_report = 0;
    g_detect_total = 0; g_red_runs = 0;
    g_last_mismatch = 0; g_last_red = 0; g_op_is_aes = 0;
}

/* Caractéristiques de l'arbre (fault_feat_t, défini dans noise_policy.h) */
static void fault_features(fault_feat_t *f)
{
    f->rs16 = g_nv.rs16; f->rs64 = g_nv.rs64; f->boots = g_nv.boots; f->gap = g_nv.gap;
    f->det = g_nv.det; f->rnd = g_nv.rnd; f->dur = g_nv.dur;
}

/* v12 : compteurs lus par l'arbre lors de la DERNIÈRE décision de politique (écrits dans le journal 'P'),
   pour que la conformité rejoue exactement l'entrée de la décision, et non l'état après le chiffrement. */
static fault_feat_t g_ff = { 0, 0, 0, 255, 0, 0, 0 };

void    fault_set_mode(uint8_t m)  { g_red_mode = (m > 2) ? 2 : m; }
uint8_t fault_get_mode(void)       { return g_red_mode; }
void    fault_set_report(uint8_t e){ g_fault_report = e ? 1 : 0; }
uint8_t fault_report_enabled(void) { return g_fault_report; }

/* v12 : en mode auto, la redondance s'active exactement quand la branche « faute » de l'arbre unique
   (policy_fault_alarm : règle de sûreté + sous-arbre appris) donne une alerte, c.-à-d. classe 2. */
uint8_t fault_redundancy_active(void)
{
    uint8_t a = 0;
    if (g_red_mode == 1) a = 1;
    else if (g_red_mode == 2) { fault_feat_t f; fault_features(&f); a = policy_fault_alarm(&f); }
    g_last_red = a;
    if (a && g_red_runs < 0xFFFFu) g_red_runs++;
    return a;
}

uint8_t fault_compare(const uint8_t *a, const uint8_t *b, uint8_t n)
{
    uint8_t diff = 0;
    for (uint8_t i = 0; i < n; i++) diff |= (uint8_t)(a[i] ^ b[i]);       /* temps constant */
    g_last_mismatch = diff ? 1 : 0;
    if (diff) {
        if (g_nv.det < FAULT_CNT_CAP) g_nv.det++;
        if (g_detect_total < 0xFFFFu) g_detect_total++;
    }
    return g_last_mismatch;
}

void noise_fault_round_error(void)
{
    g_round_err_op = 1;
    if (g_nv.rnd < FAULT_CNT_CAP) g_nv.rnd++;
    if (g_round_err_total < 0xFFFFu) g_round_err_total++;
}
void    fault_op_begin(void)              { g_round_err_op = 0; g_op_is_aes = 1; }
uint8_t fault_round_error_this_op(void)   { return g_round_err_op; }

void fault_fill_signals(uint8_t *b)
{
    if (!b) return;
    uint32_t n = (g_last_net_cycles > 0xFFFFFFu) ? 0xFFFFFFu : g_last_net_cycles;
    uint32_t t = (g_last_enc_cycles > 0xFFFFFFu) ? 0xFFFFFFu : g_last_enc_cycles;
    b[0] = (uint8_t)(n >> 16); b[1] = (uint8_t)(n >> 8); b[2] = (uint8_t)n;
    b[3] = (uint8_t)(t >> 16); b[4] = (uint8_t)(t >> 8); b[5] = (uint8_t)t;
    b[6] = g_nv.rnd;
    b[7] = (uint8_t)((g_round_err_op ? 0x01 : 0) | (g_last_mismatch ? 0x02 : 0) | (g_last_red ? 0x04 : 0));
}

void fault_on_operation_done(void)
{
    /* v12 : anomalie de durée nette, seulement pour un AES exécuté en mode politique (niveaux de même masque) */
    if (g_op_is_aes && g_adaptive_enabled && g_last_net_cycles > 0u) {
        uint32_t d = (g_last_net_cycles > FAULT_NET_REF) ? g_last_net_cycles - FAULT_NET_REF
                                                         : FAULT_NET_REF - g_last_net_cycles;
        if (d > FAULT_NET_TOL && g_nv.dur < FAULT_CNT_CAP) g_nv.dur++;
    }
    g_op_is_aes = 0;
    if (g_nv.gap < 255u) g_nv.gap++;
    g_nv.ops++;
    if ((g_nv.ops & 0x0Fu) == 0u) g_nv.rs16 >>= 1;                 /* toutes les 16 opérations  */
    if ((g_nv.ops & 0x3Fu) == 0u) g_nv.rs64 >>= 1;                 /* toutes les 64 opérations  */
    if ((g_nv.ops & 0xFFu) == 0u) {                                /* toutes les 256 opérations */
        g_nv.det >>= 1; g_nv.rnd >>= 1; g_nv.boots >>= 1; g_nv.dur >>= 1;
    }
}

void fault_fill_health(uint8_t *b)
{
    if (!b) return;
    put_be16(b, g_nv.det);
    b[2] = g_nv.boots;
    b[3] = (uint8_t)((g_last_mismatch ? 0x01 : 0) | (g_last_red ? 0x02 : 0) | (g_cold_boot ? 0x04 : 0)
                     | ((g_reset_cause & 0x0Fu) << 4));
    put_be16(b + 4, g_detect_total);
    put_be16(b + 6, g_red_runs);
}

void fault_clear_counters(void)
{
    nv_zero();
    g_round_err_total = 0; g_round_err_op = 0;
    g_detect_total = 0; g_red_runs = 0;
    g_last_mismatch = 0; g_last_red = 0;
}

/* Cible de faute : contrôle d'autorisation. En fonctionnement normal, l'entrée diffère de la référence ->
   accès refusé. Un glitch qui fait « sauter » la branche accorde l'accès : c'est un CONTOURNEMENT.
   Les deux variables sont volatiles : le compilateur ne peut pas supprimer la comparaison. */
static volatile uint32_t g_auth_in  = 0x00001234u;
static volatile uint32_t g_auth_ref = 0x00005678u;

uint32_t fault_check_once(void)
{
    volatile uint32_t r = 0;
    if (g_auth_in != g_auth_ref) { r = FAULT_DENIED; }
    else                         { r = FAULT_GRANTED; }
    return r;
}

/* Cible de faute n°2 : boucle imbriquée de comptage (cible classique du tutoriel ChipWhisperer Fault 1.1).
   Fenêtre de vulnérabilité de plusieurs milliers de cycles : un glitch perturbe un compteur ou un branchement
   de boucle, donc cnt != 40*40. Une valeur différente = faute induite = CONTOURNEMENT (accès accordé). */
#define LOOP_N 40
uint32_t fault_loop_once(void)
{
    volatile uint32_t cnt = 0;
    volatile uint32_t i, j;
    for (i = 0; i < LOOP_N; i++)
        for (j = 0; j < LOOP_N; j++)
            cnt++;
    return (cnt != (uint32_t)(LOOP_N * LOOP_N)) ? FAULT_GRANTED : FAULT_DENIED;
}

/* Les zones de bruit qui précèdent l'opération sensible : mêmes points d'injection que l'AES (ronde 1) ; le
   trigger est déjà levé, donc le bruit décale l'opération par rapport au trigger. */
void noise_fault_gate(void)
{
    uint8_t dummy_order[4] = {0, 1, 2, 3};
    protect_zone(ZONE_ROUND_START, 1, 0);
    protect_zone(ZONE_SUBBYTES_PRE, 1, dummy_order);
    protect_zone(ZONE_MIXCOLUMNS, 1, 0);
    protect_zone(ZONE_ADDROUNDKEY_PRE, 1, 0);
}

/* ---------------------------------------------------------------------- */
/* Mélange (shuffling) v8                                                  */
/* ---------------------------------------------------------------------- */
uint8_t noise_rand8(void) { return (uint8_t)rng_next(); }

static uint8_t g_last_order_active = 0;     /* v10 : le dernier appel à noise_get_order() a-t-il mélangé ? */
uint8_t noise_last_order_active(void) { return g_last_order_active; }

uint8_t noise_get_order(injection_zone_t zone, uint8_t round, uint8_t *order,
                        uint8_t n_real, uint8_t dummy_scale, uint8_t max_dummy)
{
    uint8_t active = 0;
    if (!(round < 16 && !(g_round_mask & (1u << round))) && (g_override_mask[zone] != 0 || g_mode_classique)) {
        decision_vec_t d = zone_decision(zone);
        active = (d.activer_mask & (1u << NOISE_PERMUTATION)) ? 1u : 0u;
    }
    g_last_order_active = active;
    uint8_t n = n_real;
    if (active && g_use_raw[NOISE_PERMUTATION] && max_dummy) {
        uint32_t dd = (uint32_t)g_raw_cycles[NOISE_PERMUTATION] * dummy_scale;
        n = (uint8_t)(n_real + ((dd > max_dummy) ? max_dummy : dd));
    }
    for (uint8_t i = 0; i < n; i++) order[i] = i;
    if (active) {
        for (uint8_t i = (uint8_t)(n - 1); i > 0; i--) {          /* Fisher-Yates */
            uint8_t j = (uint8_t)(rng_next() % (uint32_t)(i + 1u));
            uint8_t t = order[i]; order[i] = order[j]; order[j] = t;
        }
        g_events_this_enc++;
        g_events_per_zone[zone]++;
    }
    return n;
}

/* ---------------------------------------------------------------------- */
/* Politique adaptative                                                    */
/* ---------------------------------------------------------------------- */
static void compute_features(uint16_t *x)
{
    uint32_t ema = mon_dt_ema >> 8;
    uint32_t dev = mon_dt_dev >> 8;
    uint32_t permille = 0;
    if (g_leak_limit_q16 > 0)
        permille = (uint32_t)(((uint64_t)g_leak_q16 * 1000u) / g_leak_limit_q16);

    x[0] = sat16(mon_n_since_key);
    x[1] = sat16(permille);
    x[2] = mon_pt_repeat;
    x[3] = sat16(mon_hd_ema >> 4);
    x[4] = (ema > 0) ? sat16((dev << 8) / ema) : 0;
    x[5] = sat16(ema);
    {   /* v12 : x6 = alerte de la branche « faute » de l'arbre unique (0/1) */
        fault_feat_t f; fault_features(&f);
        x[6] = policy_fault_alarm(&f);
    }
}

static void apply_level(uint8_t level)
{
    const policy_level_t *L = &POLICY_LEVELS[level];
    noise_clear_all_overrides();
    noise_clear_raw_intensity();
    for (uint8_t g = 1; g < NOISE_TYPE_COUNT; g++)
        if (L->raw[g] > 0) noise_set_raw_intensity((noise_type_t)g, L->raw[g]);
    for (uint8_t i = 0; i < L->n_forces && i < POLICY_MAX_FORCES; i++)
        noise_force_generator((injection_zone_t)L->forces[i].zone, (noise_type_t)L->forces[i].gen, INTENSITY_LOW);
    g_round_mask = L->round_mask;
    g_applied_level = level;
}

void noise_decide_and_apply(void)
{
    if (!g_adaptive_enabled) { g_cur_level = 0xFF; return; }
    uint32_t t0 = DWT_CYCCNT;

    compute_features(g_x);
    fault_features(&g_ff);
    uint8_t level = policy_decide(g_x, &g_ff, g_leak_q16, g_leak_limit_q16, &g_cur_class);
    if (level >= POLICY_N_LEVELS) level = POLICY_N_LEVELS - 1;
    if (level != g_applied_level) apply_level(level);
    g_cur_level = level;

    g_decide_cycles = sat16(DWT_CYCCNT - t0);
}

void noise_set_leak_limit(uint32_t limit_q16) { g_leak_limit_q16 = limit_q16; }

uint8_t noise_policy_fill_log(uint8_t *b)
{
    if (!b) return 0;
    b[0] = g_cur_level;
    put_be16(b + 1, g_x[0]); put_be16(b + 3, g_x[1]);                 /* x0, x1 : octets 1-4 */
    {   const fault_feat_t *f = &g_ff;                                 /* v12 : octets 5-12 = compteurs de la décision */
        b[5] = f->rs16; b[6] = f->rs64; b[7] = f->boots; b[8] = f->gap;
        b[9] = f->det;  b[10] = f->rnd; b[11] = f->dur;  b[12] = 0; }
    put_be16(b + 13, g_decide_cycles);
    b[15] = (uint8_t)(((g_leak_limit_q16 > 0 && g_leak_q16 >= g_leak_limit_q16) ? 0x01 : 0x00)
                      | ((g_x[6] & 0x07u) << 1)                             /* v10.1 : x6 sur 3 bits */
                      | ((g_cur_class & 0x07u) << 4));
    return POLICY_LOG_LEN;
}

void noise_enable_adaptive(uint8_t e)
{
    g_adaptive_enabled = e ? 1 : 0;
    g_applied_level = 0xFF;
    g_cur_level = 0xFF;
    if (!e) { noise_clear_all_overrides(); noise_clear_raw_intensity(); g_round_mask = ROUND_MASK_ALL; }
}
uint8_t noise_is_adaptive(void) { return g_adaptive_enabled; }

/* ---------------------------------------------------------------------- */
/* Cycle de vie d'un chiffrement                                           */
/* ---------------------------------------------------------------------- */
void noise_generator_init(void)
{
    DEMCR |= (1u << 24);
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;

    rng_seed();                         /* v10 : graine ADC (repli : graine fixe), re-mélangée à chaque plaintext */
    g_fm_shift = 2;                     /* v10 : délais à moyenne flottante par défaut */
    g_mode_classique = 0;
    g_fixed_intensity = INTENSITY_MEDIUM;
    g_logging_enabled = 0;
    g_round_mask = ROUND_MASK_ALL;
    for (uint8_t r = 0; r < NOISE_MAX_ROUNDS; r++)
        for (uint8_t z = 0; z < ZONE_COUNT; z++) g_risk_map[r][z] = 77;

    noise_clear_all_overrides();
    noise_clear_raw_intensity();
    monitor_reset();
    fault_mon_init();
    g_adaptive_enabled = 0;
    g_applied_level = 0xFF;
    g_cur_level = 0xFF;
    g_leak_limit_q16 = POLICY_LEAK_LIMIT_Q16;
}

void noise_begin_encryption(void)
{
    g_fm_draw = rng_next();                                 /* v10 : moyenne des délais de CE chiffrement */
    g_events_this_enc = 0;
    g_total_event_cycles = 0;
    for (uint8_t z = 0; z < ZONE_COUNT; z++) g_events_per_zone[z] = 0;
    if (g_ts_state == 1) {                                  /* armé : on enregistre ce chiffrement */
        memset(g_zone_ts, 0xFF, sizeof(g_zone_ts));
        g_ts_state = 2;
    }
    g_enc_start_cycle = DWT_CYCCNT;                         /* en dernier : origine des horodatages */
}

void noise_end_encryption(void)
{
    g_last_enc_cycles = DWT_CYCCNT - g_enc_start_cycle;   /* v10.1 : en premier, avant tout calcul */
    g_last_net_cycles = (g_last_enc_cycles > g_total_event_cycles) ? g_last_enc_cycles - g_total_event_cycles : 0;
    if (g_ts_state == 2) g_ts_state = 0;
    if (g_adaptive_enabled && g_cur_level < POLICY_N_LEVELS) {
        g_leak_q16 = sat_add32(g_leak_q16, POLICY_LEVELS[g_cur_level].leak_q16);
        bud_save();
    }
}

void noise_dump_summary(void)
{
    if (!g_logging_enabled) return;
    putch('S'); putch(' ');
    putch('e'); putch('='); putc_hex16(g_events_this_enc); putch(' ');
    putch('c'); putch('='); putc_hex24(g_total_event_cycles);
    for (uint8_t z = 0; z < ZONE_COUNT; z++) {
        putch(' '); putch('z'); putch('0' + z); putch('='); putc_hex16(g_events_per_zone[z]);
    }
    putch(' '); putch('m'); putch('='); putc_hex16(g_round_mask);
    putch('\n');
}

/* ---------------------------------------------------------------------- */
/* Horodatage des zones                                                    */
/* ---------------------------------------------------------------------- */
void noise_arm_zone_timestamps(void) { g_ts_state = 1; }

void noise_get_zone_timestamps(uint8_t round, uint8_t *b)
{
    for (uint8_t z = 0; z < ZONE_COUNT; z++)
        put_be32(b + 4 * z, (round < NOISE_MAX_ROUNDS) ? g_zone_ts[round][z] : 0xFFFFFFFFu);
}

/* ---------------------------------------------------------------------- */
/* Setters                                                                  */
/* ---------------------------------------------------------------------- */
void noise_set_mode(uint8_t m) { g_mode_classique = m ? 1 : 0; }
void noise_set_fixed_intensity(noise_intensity_t i) { if (i < INTENSITY_COUNT) g_fixed_intensity = i; }
void noise_set_logging(uint8_t e) { g_logging_enabled = e ? 1 : 0; }
void noise_set_per_event_log(uint8_t e) { g_per_event_log = e ? 1 : 0; }
void noise_set_round_mask(uint16_t m) { g_round_mask = m & ROUND_MASK_ALL; }
uint16_t noise_get_round_mask(void) { return g_round_mask; }

void noise_set_risk(uint8_t r, injection_zone_t z, uint8_t v)
{ if (r < NOISE_MAX_ROUNDS && z < ZONE_COUNT) g_risk_map[r][z] = v; }
void noise_set_generator_weight(noise_type_t g, uint8_t w) { if (g < NOISE_TYPE_COUNT) g_generator_weight[g] = w; }
void noise_set_budget_cycles(uint16_t b) { g_budget_cycles_per_zone = b; }

void noise_force_generator(injection_zone_t z, noise_type_t g, noise_intensity_t i)
{
    if (z >= ZONE_COUNT || g >= NOISE_TYPE_COUNT || i >= INTENSITY_COUNT) return;
    g_override_mask[z] |= (uint8_t)(1u << g);
    g_override_intensity[z][g] = i;
}
void noise_clear_override(injection_zone_t z) { if (z < ZONE_COUNT) g_override_mask[z] = 0; }
void noise_clear_all_overrides(void) { for (uint8_t z = 0; z < ZONE_COUNT; z++) g_override_mask[z] = 0; }

void noise_get_current_config(uint8_t *zone, uint8_t *gen, uint8_t *intensity)
{
    for (uint8_t z = 0; z < ZONE_COUNT; z++) {
        if (g_override_mask[z] != 0) {
            *zone = z;
            for (uint8_t g = 1; g < NOISE_TYPE_COUNT; g++)
                if (g_override_mask[z] & (1u << g)) {
                    *gen = g;
                    *intensity = (uint8_t)g_override_intensity[z][g];
                    return;
                }
        }
    }
    *zone = 0xFF; *gen = 0; *intensity = 0;
}

void noise_get_observation(uint8_t *obs, uint8_t len)
{
    uint8_t tmp[POLICY_LOG_LEN];
    uint16_t x[POLICY_N_FEATURES];
    compute_features(x);
    tmp[0] = g_cur_level;
    put_be16(tmp + 1, x[0]); put_be16(tmp + 3, x[1]);
    {   fault_feat_t f; fault_features(&f);
        tmp[5] = f.rs16; tmp[6] = f.rs64; tmp[7] = f.boots; tmp[8] = f.gap;
        tmp[9] = f.det;  tmp[10] = f.rnd; tmp[11] = f.dur;  tmp[12] = 0; }
    put_be16(tmp + 13, g_decide_cycles);
    tmp[15] = (uint8_t)(((g_leak_limit_q16 > 0 && g_leak_q16 >= g_leak_limit_q16) ? 0x01 : 0x00)
                        | ((x[6] & 0x07u) << 1)
                        | ((g_cur_class & 0x07u) << 4));
    for (uint8_t i = 0; i < len && i < POLICY_LOG_LEN; i++) obs[i] = tmp[i];
}

void noise_set_raw_intensity(noise_type_t g, uint16_t c)
{ if (g >= NOISE_TYPE_COUNT) return; g_raw_cycles[g] = c; g_use_raw[g] = (c > 0) ? 1 : 0; }
void noise_clear_raw_intensity(void)
{ for (uint8_t g = 0; g < NOISE_TYPE_COUNT; g++) { g_raw_cycles[g] = 0; g_use_raw[g] = 0; } }
uint16_t noise_get_raw_intensity(noise_type_t g) { return (g >= NOISE_TYPE_COUNT) ? 0 : g_raw_cycles[g]; }
uint8_t  noise_is_raw_active(noise_type_t g)     { return (g >= NOISE_TYPE_COUNT) ? 0 : g_use_raw[g]; }

/* ---------------------------------------------------------------------- */
/* Points d'injection                                                       */
/* ---------------------------------------------------------------------- */
void noise_zone_round_start(uint8_t r)                          { protect_zone(ZONE_ROUND_START, r, 0); }
void noise_zone_subbytes_pre(uint8_t o[4], uint8_t r)           { protect_zone(ZONE_SUBBYTES_PRE, r, o); }
void noise_zone_subbytes_byte(volatile uint8_t *b, uint8_t r)   { protect_zone(ZONE_SUBBYTES_BYTE, r, (void *)b); }
void noise_zone_mixcolumns(uint8_t r)                           { protect_zone(ZONE_MIXCOLUMNS, r, 0); }
void noise_zone_addroundkey_pre(uint8_t r)                      { protect_zone(ZONE_ADDROUNDKEY_PRE, r, 0); }
void noise_zone_addroundkey_byte(volatile uint8_t *b, uint8_t r){ protect_zone(ZONE_ADDROUNDKEY_BYTE, r, (void *)b); }
void noise_zone_round_end(uint8_t r)                            { protect_zone(ZONE_ROUND_END, r, 0); }