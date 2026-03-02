# 🛒 Projet ML Retail — Analyse Comportementale Clientèle

> Atelier Machine Learning · Module GI2 · Préparé par Fadoua Drira · 2025-2026

## 📋 Description

Pipeline complet de Machine Learning sur un dataset e-commerce de cadeaux (52 features, 2 000 clients).  
L'objectif est de comprendre le comportement client, prédire le churn et segmenter la clientèle.

**Tâches ML couvertes :**
- **Clustering** — K-Means avec sélection du k optimal (Elbow + Silhouette)
- **ACP** — Réduction dimensionnelle (variance expliquée ≥ 95%)
- **Classification** — Prédiction du Churn (Random Forest + Gradient Boosting)
- **Régression** — Prédiction de MonetaryTotal (Random Forest Regressor)

---

## 🗂️ Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/                   # Données brutes 
│   │   └── retail.csv
│   ├── processed/             # Données nettoyées
│   │   └── projet_processed.csv
│   └── train_test/            # Données splittées
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
├── notebooks/                 # Notebooks Jupyter (prototypage)
├── src/
│   ├── utils.py               # Fonctions utilitaires partagées
│   ├── preprocessing.py       # Pipeline de préparation des données
│   ├── train_model.py         # Entraînement des 4 modèles ML
│   └── predict.py             # Prédictions sur nouvelles données
├── models/                    # Modèles sauvegardés (.joblib)
├── app/
│   ├── app.py                 # Application Flask
│   └── templates/             # HTML (index, result, dashboard)
├── reports/                   # Visualisations et rapports générés
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/mohemed-amine-gharbi/projet_ml_retail
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
# Création
python -m venv venv

# Activation — Windows
venv\Scripts\activate

# Activation — Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Étape 1 — Exploration le dataset

excuter notebooks/exploration.ipynb


### Étape 2 — Prétraitement
```bash
python src/preprocessing.py
```
Pipeline complet : nettoyage → parsing → imputation → encodage → feature engineering → split train/test.

### Étape 3 — Entraînement des modèles
```bash
python src/train_model.py
```
Entraîne et sauvegarde les 4 modèles dans `models/`. Génère les visualisations dans `reports/`.

### Étape 4 — Prédictions
```bash
python src/predict.py --mode all       # Tous les modèles
python src/predict.py --mode churn     # Churn uniquement
python src/predict.py --mode segment   # Segmentation uniquement
```

### Étape 5 — Application Web Flask
```bash
cd app
python app.py
```
Ouvrez votre navigateur sur **http://127.0.0.1:5000**

---

## 📊 Résultats & Rapports

Après entraînement, les fichiers suivants sont générés dans `reports/` :

| Fichier | Description |
|---|---|
| `pca_analysis.png` | Courbe de variance ACP + projection 2D |
| `kmeans_selection.png` | Méthode Elbow + Score Silhouette |
| `kmeans_clusters.png` | Visualisation des clusters en 2D |
| `cluster_profiles.csv` | Profil moyen par cluster |
| `classification_results.png` | Courbe ROC + matrice de confusion |
| `feature_importance.png` | Importance des features (Random Forest) |
| `regression_results.png` | Réel vs Prédit + distribution résidus |
| `churn_predictions.csv` | Prédictions churn sur X_test |
| `segment_predictions.csv` | Segments prédits sur X_test |

---

## 🧩 Problèmes de qualité traités

| Problème | Feature(s) | Traitement |
|---|---|---|
| Valeurs manquantes (30%) | Age | Imputation médiane |
| Valeurs aberrantes | SupportTickets (-1, 999) | Détection → NaN → médiane |
| Valeurs aberrantes | Satisfaction (-1, 99) | Détection → NaN → médiane |
| Formats inconsistants | RegistrationDate | Parsing pandas to_datetime |
| Feature constante | NewsletterSubscribed | Suppression |
| Feature brute | LastLoginIP | Extraction : IP_IsPrivate, IP_FirstOctet |
| Déséquilibre classes | Churn (~25% positif) | class_weight='balanced' |

---

## 📦 Dépendances principales

```
numpy
pandas
scikit-learn
matplotlib
seaborn
flask
joblib
```

---

## 🔗 API REST

L'application expose un endpoint JSON :

```bash
POST /api/predict
Content-Type: application/json

{
  "Recency": 45,
  "Frequency": 3,
  "MonetaryTotal": 250
}
```

**Réponse :**
```json
{
  "churn_probability": 0.7234,
  "churn_predicted": 1,
  "risk_level": "Élevé",
  "segment": 2
}
```
