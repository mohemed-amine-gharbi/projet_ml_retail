# Projet Machine Learning – Retail

## Description

Ce projet a pour objectif de développer un pipeline complet de Machine Learning appliqué à des données retail (commerce).

Il inclut :

- Le nettoyage et la préparation des données
- L'entraînement d’un modèle de Machine Learning
- La sauvegarde du modèle
- La génération de prédictions
- Une application web (Flask) pour exploiter le modèle
- Une organisation professionnelle du code

Le projet est structuré selon les bonnes pratiques en Data Science et ML Engineering.

---

## Structure du projet

projet_ml_retail/
│
├── data/
│ ├── raw/ # Données brutes originales
│ ├── processed/ # Données nettoyées
│ └── train_test/ # Données splittées (train / test)
│
├── notebooks/ # Notebooks Jupyter (prototypage)
│
├── src/ # Scripts Python (production)
│ ├── preprocessing.py
│ ├── train_model.py
│ ├── predict.py
│ └── utils.py
│
├── models/ # Modèles sauvegardés (.pkl / .joblib)
│
├── app/ # Application web (Flask)
│
├── reports/ # Rapports et visualisations
│
├── requirements.txt # Dépendances du projet
├── README.md # Documentation
└── .gitignore


---

## Installation

### 1. Cloner le projet

git clone https://github.com/mohemed-amine-gharbi/projet_ml_retail.git

---

### 2. Créer un environnement virtuel

python -m venv venv

---

### 3. Activer l’environnement virtuel

.\venv\Scripts\Activate

---

### 4. Installer les dépendances

pip install -r requirements . txt

---

## Utilisation du projet

### 1. Prétraitement des données


Cette étape :
- Nettoie les données
- Effectue les transformations nécessaires
- Sauvegarde les données préparées dans `data/processed/`

---

### 2. Entraînement du modèle


Cette étape :
- Charge les données préparées
- Entraîne le modèle
- Sauvegarde le modèle dans `models/`

---

### 3. Générer des prédictions


---

### 4. Lancer l’application Flask


Puis ouvrir dans le navigateur : http://127.0.0.1:5000/


---

## Dépendances principales

Les principales bibliothèques utilisées peuvent inclure :

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- flask  
- joblib  

La liste complète des dépendances est disponible dans `requirements.txt`.

---

## Bonnes pratiques suivies

- Séparation claire entre données brutes et données traitées
- Séparation du code de production et du prototypage
- Sauvegarde des modèles
- Environnement virtuel isolé
- Gestion des dépendances via `requirements.txt`
- Versionnement avec Git

---

## Auteur

Projet réalisé dans le cadre d’un projet académique en Machine Learning.
