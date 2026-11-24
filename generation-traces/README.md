# Régénération des traces SCA

Les campagnes brutes (≈ 800 Go) ne sont pas incluses dans ce dépôt : elles
sont stockées sur disque externe et peuvent être régénérées avec les scripts
de `projet/`. Voir le mémoire, section 5.5 (« Méthodologie d'exploration »).

## Prérequis matériels

- ChipWhisperer-Lite + carte CW308T-STM32F3 (ou CW-Lite + cible STM32F3)
- Firmware compilé : `firmware/mcu6/simpleserial-aes/simpleserial-aes-CWLITEARM.hex`
- Disque externe monté sur `/media/khadim/2A53DBCE3700DFDA` (configurable
  dans `projet/banc_v10.py`, variable `DISQUE`)

## Prérequis logiciels

- Python 3 avec `numpy`, `scipy`, `pandas`, `scikit-learn`, `chipwhisperer`, `pycryptodome`
- ChipWhisperer SDK installé et fonctionnel (`python -c "import chipwhisperer"`)

## Campagnes du chapitre 6

| Campagne | Script | Nombre de traces | But |
|---|---|---|---|
| Référence sans bruit | `projet/banc_v10.py` (capturer) | 20 000 | Référence propre (base_v10.json) |
| Délais aléatoires | `projet/banc_v10.py` (capturer) | 20 000 | Tableau 6.1 |
| Niveau 3 | `projet/grande_v10.py` | 50 000 | Section 6.2 |
| Glitchs d'horloge | `projet/glitch_v10.py` | 3 000 × 3 modes | Section 6.4 |
| Sessions normales | `projet/sessions_v10.py` | 600 × 4 scénarios | Section 6.4 |

## Vérification d'intégrité

L'empreinte du firmware utilisé doit correspondre à celle indiquée dans
`VERSIONS.md` :

    sha256sum firmware/mcu6/simpleserial-aes/simpleserial-aes-CWLITEARM.hex

## Organisation des traces générées

    /media/khadim/2A53DBCE3700DFDA/trace/v10/
    ├── base_v10.json                  ← référence sans bruit
    ├── calibration_v10.json
    ├── campagnes/                     ← campagnes du chapitre 6
    ├── validation/                    ← campagnes 20k pour la validation
    ├── validation/grande/             ← campagne 50k
    ├── parego/                        ← historique d'optimisation
    └── fautes/                        ← glitchs et sessions normales
