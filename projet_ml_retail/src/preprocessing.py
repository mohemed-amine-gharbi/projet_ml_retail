# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
preprocessing.py — Pipeline complet de préparation des données
Projet ML Retail — Analyse Comportementale Clientèle

Étapes :
1. Chargement des données brutes
2. Suppression des features inutiles
3. Parsing (RegistrationDate, LastLoginIP)
4. Correction des valeurs aberrantes
5. Imputation des valeurs manquantes
6. Feature Engineering
7. Encodage des variables catégorielles
8. Suppression des features fortement corrélées
9. Split Train/Test + Normalisation + sauvegarde
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (load_raw_data, summarize_dataframe,
                   DATA_PROCESSED, DATA_TRAIN_TEST, MODELS_DIR, save_model)


# ── 1. Chargement ────────────────────────────────────────────────────────────
def load_and_inspect(filename="retail.csv"):
    df = load_raw_data(filename)
    summarize_dataframe(df)
    return df


# ── 2. Suppression des features inutiles ────────────────────────────────────
def drop_useless_features(df):
    """Supprime les colonnes sans valeur prédictive."""
    to_drop = []
    for col in df.columns:
        if df[col].nunique() == 1:
            to_drop.append(col)
            print(f"  🗑️  Supprimé (variance nulle) : {col}")
    if "CustomerID" in df.columns:
        to_drop.append("CustomerID")
        print(f"  🗑️  Supprimé (identifiant) : CustomerID")
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    return df


# ── 3. Parsing RegistrationDate & LastLoginIP ────────────────────────────────
def parse_special_features(df):
    """Parse les features brutes en features exploitables."""
    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"], dayfirst=True, errors="coerce"
        )
        df["RegYear"]    = df["RegistrationDate"].dt.year.fillna(0).astype(int)
        df["RegMonth"]   = df["RegistrationDate"].dt.month.fillna(0).astype(int)
        df["RegDay"]     = df["RegistrationDate"].dt.day.fillna(0).astype(int)
        df["RegWeekday"] = df["RegistrationDate"].dt.weekday.fillna(-1).astype(int)
        df = df.drop(columns=["RegistrationDate"])
        print("  ✅ RegistrationDate → RegYear, RegMonth, RegDay, RegWeekday")

    if "LastLoginIP" in df.columns:
        def is_private_ip(ip):
            try:
                p = [int(x) for x in str(ip).split(".")]
                return 1 if (p[0]==10 or (p[0]==172 and 16<=p[1]<=31)
                             or (p[0]==192 and p[1]==168)) else 0
            except Exception:
                return -1

        def ip_first_octet(ip):
            try:
                return int(str(ip).split(".")[0])
            except Exception:
                return -1

        df["IP_IsPrivate"]  = df["LastLoginIP"].apply(is_private_ip)
        df["IP_FirstOctet"] = df["LastLoginIP"].apply(ip_first_octet)
        df = df.drop(columns=["LastLoginIP"])
        print("  ✅ LastLoginIP → IP_IsPrivate, IP_FirstOctet")
    return df


# ── 4. Correction des valeurs aberrantes ────────────────────────────────────
def fix_outliers(df):
    """Corrige les valeurs aberrantes identifiées."""
    if "SupportTickets" in df.columns:
        n = ((df["SupportTickets"] < 0) | (df["SupportTickets"] == 999)).sum()
        df["SupportTickets"] = df["SupportTickets"].replace(-1, np.nan)
        df["SupportTickets"] = df["SupportTickets"].where(df["SupportTickets"] <= 15, np.nan)
        print(f"  ✅ SupportTickets : {n} aberrations → NaN")
    if "Satisfaction" in df.columns:
        n = ((df["Satisfaction"] < 0) | (df["Satisfaction"] > 5)).sum()
        df["Satisfaction"] = df["Satisfaction"].where(
            (df["Satisfaction"] >= 0) & (df["Satisfaction"] <= 5), np.nan)
        print(f"  ✅ Satisfaction : {n} aberrations → NaN")
    return df


# ── 5. Imputation des valeurs manquantes ────────────────────────────────────
def impute_missing(df):
    """Impute les valeurs manquantes selon la nature des features."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if "Age" in df.columns:
        med = df["Age"].median()
        df["Age"] = df["Age"].fillna(med)
        print(f"  ✅ Age imputé par médiane : {med:.1f}")

    for col in ["SupportTickets", "Satisfaction"]:
        if col in df.columns and df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"  ✅ {col} imputé par médiane : {med:.1f}")

    remaining_nan = [c for c in num_cols if df[c].isnull().any()]
    if remaining_nan:
        print(f"  ⚙️  KNN Imputation sur : {remaining_nan}")
        knn = KNNImputer(n_neighbors=5)
        df[remaining_nan] = knn.fit_transform(df[remaining_nan])

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Inconnu")
    return df


# ── 6. Feature Engineering ───────────────────────────────────────────────────
def feature_engineering(df):
    """Crée de nouvelles features à partir des existantes."""
    if "MonetaryTotal" in df.columns and "Recency" in df.columns:
        df["MonetaryPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasketValue"] = df["MonetaryTotal"] / df["Frequency"].replace(0, 1)
    if "Recency" in df.columns and "CustomerTenure" in df.columns:
        df["TenureRatio"] = df["Recency"] / (df["CustomerTenure"] + 1)
    if "CancelledTrans" in df.columns and "TotalTrans" in df.columns:
        df["CancelRate"] = df["CancelledTrans"] / (df["TotalTrans"] + 1)
    if "UniqueProducts" in df.columns and "TotalTrans" in df.columns:
        df["ProductsPerTrans"] = df["UniqueProducts"] / (df["TotalTrans"] + 1)
    print("  ✅ Feature Engineering : 5 nouvelles features créées")
    return df


# ── 7. Encodage des variables catégorielles ──────────────────────────────────
def encode_categoricals(df):
    """Encode les variables catégorielles selon leur type."""

    # Dictionnaire large pour couvrir les noms de colonnes variés du vrai dataset
    ordinal_mappings = {
        # Noms générés
        "SpendingCat":      ["Low", "Medium", "High", "VIP"],
        "AgeCategory":      ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"],
        "LoyaltyLevel":     ["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"],
        "ChurnRisk":        ["Faible", "Moyen", "Élevé", "Critique"],
        "PreferredTime":    ["Nuit", "Matin", "Midi", "Après-midi", "Soir"],
        "BasketSize":       ["Petit", "Moyen", "Grand", "Inconnu"],
        # Noms alternatifs du vrai dataset
        "SpendingCategory": ["Low", "Medium", "High", "VIP"],
        "PreferredTimeOfDay": ["Night", "Morning", "Noon", "Afternoon", "Evening",
                               "Nuit", "Matin", "Midi", "Après-midi", "Soir"],
        "ChurnRiskCategory":["Low", "Medium", "High", "Critical",
                             "Faible", "Moyen", "Élevé", "Critique"],
        "WeekendPreference":["Weekday", "Weekend", "Unknown",
                             "Semaine", "Weekend", "Inconnu"],
        "BasketSizeCategory":["Small", "Medium", "Large", "Unknown",
                              "Petit", "Moyen", "Grand", "Inconnu"],
    }

    for col, order in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: order.index(x) if x in order else len(order) // 2
            )
            print(f"  ✅ Encodage ordinal : {col}")

    # One-Hot Encoding — noms générés ET alternatifs
    one_hot_candidates = [
        "CustomerType", "FavoriteSeason", "Region",
        "WeekendPref", "ProdDiversity", "ProductDiverssity",
        "Gender", "AccountStatus", "RFMSegment",
    ]
    one_hot_cols = [c for c in one_hot_candidates if c in df.columns]
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False, dtype=int)
        print(f"  ✅ One-Hot Encoding : {one_hot_cols}")

    # Frequency encoding pour Country (haute cardinalité)
    if "Country" in df.columns:
        country_counts = df["Country"].value_counts()
        df["Country_Freq"] = df["Country"].map(country_counts)
        df = df.drop(columns=["Country"])
        print("  ✅ Country → Country_Freq (frequency encoding)")

    # LabelEncoder pour toute colonne objet restante
    remaining_obj = df.select_dtypes(include=["object"]).columns.tolist()
    # Exclure les colonnes cible
    remaining_obj = [c for c in remaining_obj if c not in ["Churn"]]
    if remaining_obj:
        print(f"  ⚙️  LabelEncoder sur colonnes résiduelles : {remaining_obj}")
        for col in remaining_obj:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


# ── 8. Suppression des features fortement corrélées ─────────────────────────
def remove_highly_correlated(df, target_col="Churn", threshold=0.90):
    """Supprime une des deux features si |corrélation| > threshold."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= threshold)]
    df = df.drop(columns=to_drop)
    if to_drop:
        print(f"  🗑️  Supprimées (corrélation > {threshold}) : {to_drop}")
    else:
        print(f"  ✅ Aucune feature supprimée pour corrélation > {threshold}")
    return df


# ── 9. Split Train/Test ──────────────────────────────────────────────────────
def split_and_save(df, target_col="Churn", test_size=0.2):
    """Sépare en train/test, normalise et sauvegarde."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Forcer tout en numérique (sécurité finale)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n  ✅ Split : {X_train.shape[0]} train | {X_test.shape[0]} test")
    print(f"     Churn train : {y_train.mean():.1%} | test : {y_test.mean():.1%}")

    # Normalisation APRÈS le split (évite le data leakage)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    save_model(scaler, "scaler")

    os.makedirs(DATA_TRAIN_TEST, exist_ok=True)
    X_train.to_csv(os.path.join(DATA_TRAIN_TEST, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_TRAIN_TEST,  "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(DATA_TRAIN_TEST, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_TRAIN_TEST,  "y_test.csv"),  index=False)
    print("  ✅ Données train/test sauvegardées dans data/train_test/")

    return X_train, X_test, y_train, y_test


# ── Pipeline complet ─────────────────────────────────────────────────────────
def run_preprocessing(filename="retail.csv"):
    print("\n" + "="*60)
    print("PIPELINE DE PRÉTRAITEMENT")
    print("="*60)

    df = load_and_inspect(filename)

    print("\n📌 Étape 1 : Suppression features inutiles")
    df = drop_useless_features(df)

    print("\n📌 Étape 2 : Parsing RegistrationDate & LastLoginIP")
    df = parse_special_features(df)

    print("\n📌 Étape 3 : Correction des valeurs aberrantes")
    df = fix_outliers(df)

    print("\n📌 Étape 4 : Imputation des valeurs manquantes")
    df = impute_missing(df)

    print("\n📌 Étape 5 : Feature Engineering")
    df = feature_engineering(df)

    print("\n📌 Étape 6 : Encodage des catégorielles")
    df = encode_categoricals(df)

    print("\n📌 Étape 7 : Suppression corrélations élevées")
    df = remove_highly_correlated(df, threshold=0.90)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    processed_path = os.path.join(DATA_PROCESSED, "projet_processed.csv")
    df.to_csv(processed_path, index=False)
    print(f"\n✅ Données traitées sauvegardées : {processed_path}")
    print(f"   Dimensions finales : {df.shape}")

    print("\n📌 Étape 8 : Split Train/Test + Normalisation")
    X_train, X_test, y_train, y_test = split_and_save(df)

    print("\n✅ Prétraitement terminé !\n")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()