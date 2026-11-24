# Générateur de bruit intelligent (AES-128, STM32F3)

Code source du mémoire de Master MS2E (UCAD, LACGAA) :
Étude et conception d'un générateur de bruit intelligent basé sur l'intelligence
artificielle pour la protection des systèmes embarqués contre les attaques par
canaux auxiliaires et par injection de fautes.

## Structure

| Dossier | Contenu |
|---|---|
| firmware/mcu6/simpleserial-aes/ | Micrologiciel C : générateur de bruit, politique embarquée, AES-128 |
| projet/ | Chaîne Python de conception : ParEGO, attaquant CPA, métriques N10, arbre de décision |
| configs/ | Fichiers de configuration (base_v10.json, levels_3niv.json) |
| donnees-legeres/ | Résultats JSON (validation, tableau final) — pas de traces brutes |
| generation-traces/ | Instructions pour régénérer les ≈ 800 Go de traces SCA |

## Installation

git clone https://github.com/Bambiste01/memoire-bruit-intelligent.git
cd memoire-bruit-intelligent

### Firmware

cd firmware/mcu6/simpleserial-aes
make PLATFORM=CWLITEARM CRYPTO_TARGET=TINYAES128C SS_VER=SS_VER_1_1

### Chaîne Python

Dépendances : numpy, scipy, pandas, scikit-learn, chipwhisperer, pycryptodome.

## Citation

Sarr, K. (2026). Générateur de bruit intelligent pour la protection des
systèmes embarqués (Version 13.0.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.24503568

## Licence

MIT — voir LICENSE.
