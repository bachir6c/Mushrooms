# Mushrooms
 Prédiction de la comestibilité des champignons avec un réseau de neurones Ce projet utilise un réseau de neurones profond (Keras/TensorFlow) pour prédire si un champignon est comestible ou vénéneux à partir de ses caractéristiques. Prétraitement, visualisation, encoding, et modélisation complète avec évaluation des performances.
 # 🍄 Prédiction de la comestibilité des champignons à l’aide d’un réseau de neurones

Ce projet a pour objectif de prédire si un champignon est **comestible** ou **vénéneux** à partir de ses caractéristiques morphologiques. Il s’appuie sur un **réseau de neurones profond** implémenté avec **TensorFlow/Keras**.

## 📌 Objectifs

- Analyser les caractéristiques des champignons du jeu de données `mushrooms.csv`
- Visualiser la distribution des variables en fonction de la toxicité
- Appliquer des techniques de **prétraitement** (encodage, standardisation)
- Construire et entraîner un **modèle de classification binaire**
- Évaluer les performances sur un jeu de test

## 🧠 Technologies utilisées

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- TensorFlow / Keras
- Réseaux de neurones profonds (Dense + Dropout)

## 📊 Étapes du projet

### 1. Chargement et exploration des données

- Chargement du jeu de données `mushrooms.csv`
- Exploration initiale (`.info()`, `.head()`, `.value_counts()`...)
- Visualisation de la distribution des caractéristiques par classe (`comestible` ou `vénéneux`)

### 2. Prétraitement des données

- Encodage de la variable cible avec `LabelEncoder`
- One-hot encoding des variables catégorielles avec `pd.get_dummies`
- Séparation des données en `X` (caractéristiques) et `y` (étiquette)
- Division du jeu de données en **train/test (80/20)**

### 3. Construction du modèle

Deux architectures ont été testées :

#### ✅ Modèle simple :

- 2 couches denses (16 puis 8 neurones) avec activation ReLU
- 1 couche de sortie (1 neurone, activation sigmoid)

#### ✅ Modèle amélioré :

- 2 couches denses (64 puis 32 neurones)
- **Dropout (0.5)** pour éviter l’overfitting
- Fonction d’activation : ReLU (couches cachées), Sigmoid (sortie)
- Optimiseur : **Adam** avec `learning_rate=0.001`
- Perte : `binary_crossentropy`

### 4. Entraînement et évaluation

- Entraînement sur 100 epochs
- Validation croisée via `validation_split` et `validation_data`
- Évaluation finale sur le jeu de test
- Affichage de la précision (`accuracy`)

### 📈 Résultat

Le modèle atteint une précision de **~99%** sur le jeu de test, grâce à une bonne séparation des classes dans les données.

## 🗃️ Jeu de données

- Source : [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)
- Variables : toutes catégorielles (ex. `cap-color`, `odor`, `gill-size`, etc.)
- Cible : `class` (e = comestible, p = vénéneux)

## 📂 Fichier de soumission

Un fichier `submission.csv` a été généré (simulé ici comme pour une compétition Kaggle), bien que non nécessaire dans ce cas.

---

## 🙋🏽 Auteur

**Mouhamed Bachir Cissé**  
Étudiant à Polytech Lyon  
Passionné par la data science, les réseaux de neurones et les applications à impact réel.

---


