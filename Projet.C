import numpy as np  # bibliothèque pour le calcul scientifique
import pandas as pd  # bibliothèque pour la manipulation et l'analyse de données!pip install python-dateutil
import matplotlib.pyplot as plt  # bibliothèque pour la visualisation de données. pyplot est un module qui fournit une interface de type MATLAB pour créer des graphiques

import seaborn as sns  # bibliothèque basée sur Matplotlib, qui simplifie la création de visualisations statistiques attrayantes. Elle intègre des fonctionnalités avancées comme des palettes de couleurs et des visualisations complexes
from sklearn.model_selection import (
    train_test_split,  # une bibliothèque pour le machine learning en Python. train_test_split est une fonction qui permet de diviser un ensemble de données en deux sous-ensembles : un pour l'entraînement et un pour le test
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam




data = pd.read_csv("/home/mouhamed-bachir.cisse/ResauxNeurones/mushrooms.csv")
print(type(data))
print(data.head())
print(data.info())
print(data.info())
print(class_distribution)

class_distribution = data["class"].value_counts()
print("Distribution des classes comestible (e) et vénéneuse (p) :")


# Afficher la distribution de chaque variable descriptive
# data.columns = les colones de data
for i in data.columns:
    print(f"\nDistribution de la variable '{i}':")
    print(data[i].value_counts())

# Exemple : Distribution de 'cap-color' en fonction de la comestibilité
plt.figure(figsize=(20, 20))
sns.countplot(data=data, x="cap-color", hue="class")
plt.title("Distribution de la couleur du chapeau ('cap-color') par classe")
plt.xlabel("Couleur du chapeau")
plt.ylabel("Nombre de champignons")
plt.legend(["Vénéneux (p)", "Comestible (e)"])
plt.show()


# Encoder la variable cible {p, e} -> {1, 0}
label_encoder = LabelEncoder()
data["class"] = label_encoder.fit_transform(data["class"])

# Afficher la répartition pour chaque variable
for column in data.columns:
    if column != 'class':  # Ne pas faire la répartition pour la colonne cible elle-même
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=column, hue='class')
        plt.title(f"Répartition de la variable '{column}' par classe (comestible vs vénéneux)")
        plt.xlabel(column)
        plt.ylabel("Nombre de champignons")
        plt.legend(['Comestible (0)', 'Vénéneux (1)'])
        plt.xticks(rotation=45)  # Pour lisibilité si les valeurs sont longues
        plt.show()

# Afficher les premières lignes pour vérifier
print(data["class"].head)

# Utiliser le One-Hot Encoding pour les variables catégorielles
data = pd.get_dummies(data)

# Afficher les premières lignes pour vérifier
print(data.head())

# Calculer la matrice de corrélation
correlation_matrice = data.corr()

# Afficher les corrélations avec la colonne 'class'
print(correlation_matrice["class"].sort_values(ascending=False))

# Calculer et afficher la matrice de corrélation
correlation_matrice = data.corr()
print(correlation_matrice["class"].sort_values(ascending=False))

# Créer une heatmap de la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrice, cmap='coolwarm', annot=False, linewidths=0.5, cbar=True, square=True)
plt.title("Matrice de corrélation des caractéristiques")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# Séparer les données en variables d'entrée (X) et étiquette de sortie (y)
X = data.drop("class", axis=1)  # Toutes les colonnes sauf la colonne 'class'
y = data["class"]  # La colonne cible 'class' qui indique comestible (0) ou vénéneux (1)

# Diviser les données en ensemble d'entraînement (80%) et ensemble de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# shape = forme(état, condition)    ;     bruises = bleus, ecraser


# Définir le modèle
model = Sequential()

# Ajouter des couches au réseau de neurones
model.add(Dense(units=16, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dense(units=8, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
# Compiler le modèle
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Entraîner le modèle
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
# Évaluer le modèle sur les données de test
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


input_dim = X_train.shape[1]  # X_train tableau de pandas qui contient mes données
# shape permet d'obtenir la taille d'un tableau
# input_shape spécifie la forme des données d'entrée
# Création d'un nouveau modèle séquentiel
model = Sequential()

# Couche d'entrée et première couche cachée
model.add(Dense(units=64, activation="relu", input_shape=(input_dim,)))
model.add(Dropout(0.5))  # Ajout d'une couche Dropout

# Deuxième couche cachée
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.5))  # Ajout d'une couche Dropout

# Couche de sortie
model.add(Dense(units=1, activation="sigmoid"))

# Compilation du modèle avec un taux d'apprentissage ajusté
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Entraînement du modèle
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
# Fichier de soumission
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": kaggle_predictions
})
submission.to_csv('submission2.csv', index=False)

print("Fichier de soumission créé avec succès.")
