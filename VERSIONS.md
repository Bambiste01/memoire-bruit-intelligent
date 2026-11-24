# Versions et empreintes

Chaque résultat du mémoire est associé à un build précis du firmware.
L'empreinte est le SHA-256 du fichier .hex, tronqué à 12 caractères.

## Tableau des versions

| Tag Git        | Firmware | SHA-256 (12 car.)   | Politique       | Utilisé dans |
|----------------|----------|---------------------|-----------------|--------------|
| v9-final       | v9       | ________________    | escalier x6     | projets2/ (avant correction) |
| v10.1          | v10.1    | ________________    | escalier x6     | sections 5.4, 6.1 |
| v13-memoire    | v13      | db31944d16f5        | arbre unique    | chapitre 6 (résultats finaux) |

## Comment calculer l'empreinte

    sha256sum firmware/mcu6/simpleserial-aes/simpleserial-aes-CWLITEARM.hex | cut -c1-12

## Fichiers de référence associés

| Fichier | Tag | SHA-256 |
|---|---|---|
| configs/base_v10.json | v10.1 | c5725793338a |
| configs/levels_3niv.json | v10.1 | b83f5b2361ab |
