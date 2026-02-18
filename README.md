# Tri Postal — Reconnaissance OCR de Codes Postaux par k-NN

Projet de Vision par Ordinateur (3e année) — Classification de chiffres manuscrits/imprimés pour la lecture automatique de codes postaux, en utilisant l'algorithme des **k plus proches voisins (k-NN)** implémenté from scratch (sans sklearn).

---

## Architecture du Projet

```
Tri_Postal_KNN/
│
├── offline/                    # Phase d'apprentissage (hors-ligne)
│   ├── utils.py                # Fonctions utilitaires (prétraitement, segmentation, features, normalisation)
│   ├── labeling.py             # Script d'entraînement (extraction features + sauvegarde modèle)
│   └── knn_data.npz            # Modèle sauvegardé (features, labels, min/max, centroïdes)
│
├── online/                     # Phase de classification (en-ligne)
│   ├── inference.py            # Script principal de prédiction (k-NN / centroïde / les deux)
│   ├── knn_utils.py            # Algorithmes de classification (k-NN, plus proche centroïde)
│   └── metrics.py              # Fonctions d'évaluation (accuracy, confusion matrix, precision, recall)
│
├── images/
│   ├── digits/                 # Images d'entraînement (0.png à 9.png, 5 exemplaires par chiffre)
│   ├── postal_code/            # Images de test (5× "59130", 5× "62487")
│   └── results/                # Résultats visuels
│
└── README.md
```

---

## Description des Modules

### `offline/utils.py` — Fonctions Utilitaires

Le cœur du projet. Contient toute la chaîne de traitement :

| Section | Fonctions | Description |
|---|---|---|
| **Prétraitement** | `binarize_img()` | Conversion BGR → Niveaux de gris → Binarisation Otsu (BINARY_INV) |
| **Segmentation** | `detect_contours()` | Détection de contours externes (RETR_EXTERNAL) |
| | `filter_contours()` | Filtrage du bruit (aire > 20px), tri par lignes puis par x |
| | `extract_characters()` | Extraction ROI → redimensionnement 64×64 → re-binarisation → correction rotation |
| | `split_touching_digits()` | Q9 : Séparation de chiffres collés par profil de projection vertical |
| **Rotation** | `correct_rotation()` | Q8 : Redressement par moments centraux (axe principal → vertical) |
| **Features** | `calculate_cavities()` | 10 features par masques de visibilité murale (N/S/E/W) |
| | `calculate_solidity()` | Solidité = aire contour / aire enveloppe convexe |
| | `create_feature_vector()` | Assemblage du vecteur final (11 features) |
| **Normalisation** | `normalize_features()` | Normalisation Min-Max sur l'ensemble d'entraînement |
| | `apply_normalization()` | Application des min/max pré-calculés sur un vecteur test |
| | `calculate_centroids()` | Calcul du vecteur moyen par classe (10 centroïdes) |
| **Données** | `save_data()` / `load_data()` | Sauvegarde/chargement du modèle (.npz) |

### `offline/labeling.py` — Script d'Entraînement

Parcourt les 10 images d'entraînement (0.png à 9.png), extrait 5 chiffres par image → **50 échantillons × 11 features**. Applique la séparation de chiffres collés (Q9), la normalisation Min-Max puis calcule les centroïdes. Sauvegarde tout dans `knn_data.npz`.

### `online/knn_utils.py` — Algorithmes de Classification

| Fonction | Description |
|---|---|
| `calculate_distance()` | Distance euclidienne entre deux vecteurs |
| `predict_features_knn()` | k-NN classique : k=3 voisins + vote majoritaire (Counter) |
| `predict_centroid_knn()` | Plus proche moyenne : distance au centroïde de chaque classe |

### `online/inference.py` — Script de Prédiction

Script principal avec sélection du mode via `--mode`. Pour chaque image de test : binarisation → segmentation → séparation chiffres collés (Q9) → extraction features (avec rotation Q8) → normalisation → prédiction → annotation visuelle → affichage des métriques.

### `online/metrics.py` — Évaluation

Calcul de : précision par chiffre, précision par code postal, matrice de confusion 10×10, precision et recall par classe.

---

## Fonctionnalités Développées

### Espace de Décision — 11 Features

Le vecteur de features combine des **descripteurs de cavités** (masques de visibilité murale) et un **descripteur de forme** (regionprops) :

| # | Feature | Description | Plage |
|---|---|---|---|
| 1 | `central_surface` | Surface de la cavité centrale (trou fermé) | [0, 1] |
| 2 | `central_nb_blocks` | Nombre de blocs connectés dans la cavité centrale / 2 | {0, 0.5, 1.0} |
| 3 | `north_surface` | Surface de la cavité ouverte vers le Nord | [0, 1] |
| 4 | `north_barycenter_y` | Barycentre Y normalisé de la cavité Nord | [0, 1] |
| 5 | `south_surface` | Surface de la cavité ouverte vers le Sud | [0, 1] |
| 6 | `south_barycenter_y` | Barycentre Y normalisé de la cavité Sud | [0, 1] |
| 7 | `east_surface` | Surface de la cavité ouverte vers l'Est | [0, 1] |
| 8 | `east_barycenter_y` | Barycentre Y normalisé de la cavité Est | [0, 1] |
| 9 | `west_surface` | Surface de la cavité ouverte vers l'Ouest | [0, 1] |
| 10 | `west_barycenter_y` | Barycentre Y normalisé de la cavité Ouest | [0, 1] |
| 11 | `solidity` | Aire contour / aire enveloppe convexe | [0, 1] |

**Principe des masques de visibilité :**
- Pour chaque pixel de fond, on détermine s'il est "bloqué" par un pixel du chiffre dans chaque direction (N, S, E, W)
- **Cavité centrale** = bloqué dans les 4 directions (trou fermé, ex: intérieur du 0, 8)
- **Cavité directionnelle** = bloqué dans 3 directions, ouvert dans 1 (ex: ouverture du 6 vers la gauche)

### Classification

| Méthode | Principe |
|---|---|
| **k-NN (k=3)** | Distance euclidienne aux 50 échantillons + vote majoritaire des 3 plus proches |
| **Plus proche centroïde** | Distance euclidienne au vecteur moyen de chaque classe (10 centroïdes) |

### Normalisation

**Min-Max Scaling** appliquée à toutes les features pour éviter que certaines dimensions dominent le calcul de distance euclidienne. Les paramètres (min, max) sont calculés sur le jeu d'entraînement et réappliqués sur les données de test.

### Recalage en Rotation (Q8)

Utilise les **moments centraux d'ordre 2** (`mu20`, `mu02`, `mu11`) pour calculer l'angle de l'axe principal :

$$\theta = \frac{1}{2} \arctan\left(\frac{2\mu_{11}}{\mu_{20} - \mu_{02}}\right)$$

- Si l'inclinaison est entre 5° et 45°, applique une contre-rotation via `cv2.getRotationMatrix2D` + `cv2.warpAffine`
- Intégré automatiquement dans `extract_characters()` → s'applique à l'entraînement et à l'inférence
- **Résultat** : corrige la confusion 3→2 (le 3 incliné ressemblait à un 2)

### Séparation de Chiffres Collés (Q9)

Détecte les rectangles contenant potentiellement plusieurs chiffres (ratio `w/h` ≥ 1.5) puis les sépare :

1. Estimation du nombre de chiffres via `round(w / h)`
2. Calcul du **profil de projection vertical** (somme des pixels blancs par colonne)
3. Recherche des **vallées** (minima locaux) comme points de séparation optimaux
4. Découpage en sous-rectangles individuels

Intégré dans `labeling.py` et `inference.py` après `filter_contours()`.

---

## Performances

### Résultats Actuels (11 features + rotation Q8 + séparation Q9)

| Méthode | Précision Chiffres | Précision Codes Postaux |
|---|---|---|
| **k-NN (k=3)** | 47/50 = **94%** | 7/10 = **70%** |
| **Centroïde** | 48/50 = **96%** | 8/10 = **80%** |

### Erreurs Résiduelles

| Confusion | Méthodes touchées | Explication |
|---|---|---|
| 5 → 7 | k-NN + Centroïde | Chiffres allongés verticalement, cavités proches |
| 7 → 1 | k-NN + Centroïde | Morphologie très similaire (traits verticaux) |
| 1 → 7 | k-NN seulement | Introduit par la rotation ; le centroïde le corrige |

### Historique des Performances

| Version | Features | k-NN Chiffres | k-NN Codes | Centroïde Chiffres | Centroïde Codes |
|---|---|---|---|---|---|
| Cavités uniquement (10 features) | 10 | 94% | 70% | 94% | 70% |
| **+ Solidité (11 features)** | **11** | **94%** | **70%** | **96%** | **80%** |
| + Excentricité + Ratio d'aspect (13 features) | 13 | 92% ↓ | 70% | 94% ↓ | 70% ↓ |
| **+ Rotation Q8 + Séparation Q9 (11 features)** | **11** | **94%** | **70%** | **96%** | **80%** |

> La version à 13 features a été abandonnée car l'excentricité et le ratio d'aspect ajoutaient du bruit (confusion 0↔8, régression 5→6).
> La rotation (Q8) corrige la confusion 3→2 mais introduit de nouvelles confusions mineures pour le k-NN (1→7, 5→7), compensées par le centroïde.

---

## Problèmes et Limites

### Problèmes Résolus

| Problème | Cause | Solution |
|---|---|---|
| **Profils de bordure inefficaces** | L'approche initiale (distance au premier pixel) ne mesurait pas les vraies cavités | Remplacement par les **masques de visibilité murale** (N/S/E/W) |
| **nb_cavities écrasé par le scaling** | Le nombre de cavités (0-2) avait une échelle incompatible avec les ratios (0-0.3) | **Normalisation Min-Max** sur toutes les features |
| **Hybride RETR_CCOMP + masques** | Mélanger deux méthodes de mesure différentes produisait des features incohérentes | Utilisation **100% masques** (approche homogène) |
| **Confusion 5→6 (centroïde)** | Cavités similaires entre 5 et 6 | Ajout de la **solidité** (Q7) — le 5 a une solidité différente du 6 |
| **Excentricité/ratio dégradent** | Redondance avec les features existantes, bruit ajouté (0↔8) | **Retirés** du vecteur final, conservés comme fonctions utilitaires |
| **Confusion 3→2** | Le 3 incliné ressemblait à un 2 | **Q8 : Recalage en rotation** (moments centraux → redressement axe principal) |

### Limites Non Résolues

| Limite | Impact | Piste d'amélioration |
|---|---|---|
| **Confusion 5↔7** | 1 erreur persistante | Ajouter des features spécifiques (angles, profils horizontaux) |
| **Confusion 7↔1** | 1 erreur persistante | Chiffres morphologiquement très proches avec nos features |
| **Petite base d'entraînement** | 50 échantillons (5 par classe) — peu de variabilité | Augmenter les données (data augmentation, rotation, bruit) |
| **Chiffres qui se touchent** | Pipeline Q9 en place, non activé sur les données actuelles | Tester sur des images avec chiffres réellement collés |
| **k fixe (k=3)** | Non optimisé, choisi empiriquement | Cross-validation pour trouver le k optimal |

---

## Commandes de Lancement

### Prérequis

```bash
pip install opencv-python numpy matplotlib
```

### 1. Entraînement (offline)

```bash
cd offline
python labeling.py
```

Sortie attendue :
```
X shape (raw): (50, 11)
y shape: (50,)
X shape (normalized): (50, 11)
Centroids shape: (10, 11)
Training data saved to knn_data.npz
```

### 2. Inférence (online)

```bash
cd online

# k-NN uniquement
python inference.py --mode features

# Centroïde uniquement
python inference.py --mode centroid

# Comparaison des deux méthodes
python inference.py --mode both
```

### 3. Pipeline complète (entraînement + inférence)

```powershell
cd offline; python labeling.py; cd ../online; python inference.py --mode both
```

---

## Constantes de Configuration

| Constante | Valeur | Fichier | Rôle |
|---|---|---|---|
| `NOISE_THRESHOLD` | 20 | `utils.py` | Aire min d'un contour pour ne pas être filtré |
| `LINE_Y_TOLERANCE` | 20 | `utils.py` | Écart Y max entre deux chiffres sur la même ligne |
| `DIGIT_SIZE` | (64, 64) | `utils.py` | Taille de redimensionnement des chiffres extraits |
| `K` | 3 | `inference.py` | Nombre de voisins pour k-NN |
| `EXPECTED_DIGITS_PER_IMAGE` | 5 | `labeling.py` | Nombre attendu de chiffres par image d'entraînement |
