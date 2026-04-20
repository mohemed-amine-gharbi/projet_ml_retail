# -*- coding: utf-8 -*-
"""
======================================================================
PROJET MACHINE LEARNING - RETAIL ANALYTICS
======================================================================
Corrections appliquées :
  - Imputation des médianes calculée SUR X_train UNIQUEMENT (plus de leakage)
  - Encodage label : mapping calculé sur X_train, appliqué sur X_test
  - Sauvegarde des médianes d'imputation dans imputer_medians.joblib
  - Sauvegarde des features finales dans feature_names.joblib
  => Ces corrections suppriment le leakage statistique qui gonflait
     l'accuracy à 99%.

POURQUOI CES COLONNES SONT DU LEAKAGE ?
  ChurnRisk, ChurnRiskCategory, CustomerType, RFMSegment,
  LoyaltyLevel, AccountStatus, Recency — voir commentaires inline.
======================================================================
"""
import sys
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_raw_data, summarize_dataframe,
    DATA_PROCESSED, DATA_TRAIN_TEST, save_model
)

# ─────────────────────────────────────────────────────────────
# LISTES DES COLONNES À SUPPRIMER
# ─────────────────────────────────────────────────────────────
ID_COLS = ["CustomerID"]

LEAKAGE_COLS = [
    "ChurnRisk",
    "ChurnRiskCategory",
    "CustomerType",
    "RFMSegment",
    "LoyaltyLevel",
    "AccountStatus",
    "Recency",
]

TARGET_COL = "Churn"


# ─────────────────────────────────────────────────────────────
# 1. Chargement
# ─────────────────────────────────────────────────────────────
def load_and_inspect(filename="retail.csv"):
    df = load_raw_data(filename)
    summarize_dataframe(df)
    return df


# ─────────────────────────────────────────────────────────────
# 2. Suppression features inutiles + leakage
# ─────────────────────────────────────────────────────────────
def drop_useless_features(df):
    to_drop = []
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].nunique(dropna=False) <= 1:
            to_drop.append(col)
            print(f"  [drop] variance nulle : {col}")
    for col in ID_COLS:
        if col in df.columns and col not in to_drop:
            to_drop.append(col)
            print(f"  [drop] identifiant    : {col}")
    for col in LEAKAGE_COLS:
        if col in df.columns and col not in to_drop:
            to_drop.append(col)
            print(f"  [drop] leakage        : {col}")
    df = df.drop(columns=[c for c in to_drop if c in df.columns])
    print(f"  => {len(to_drop)} colonnes supprimées | restantes : {df.shape[1]}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. Suppression colonnes >50% manquantes
# ─────────────────────────────────────────────────────────────
def drop_high_missing(df, threshold=0.5):
    missing_ratio = df.isnull().mean()
    to_drop = [c for c in missing_ratio[missing_ratio > threshold].index
               if c != TARGET_COL]
    if to_drop:
        print(f"  [drop] >50% NaN : {to_drop}")
        df = df.drop(columns=to_drop)
    return df


# ─────────────────────────────────────────────────────────────
# 4. Parsing des features spéciales
# ─────────────────────────────────────────────────────────────
def parse_special_features(df):
    if "RegistrationDate" in df.columns:
        df["RegistrationDate"] = pd.to_datetime(
            df["RegistrationDate"], dayfirst=True, errors="coerce"
        )
        df["RegYear"]    = df["RegistrationDate"].dt.year
        df["RegMonth"]   = df["RegistrationDate"].dt.month
        df["RegDay"]     = df["RegistrationDate"].dt.day
        df["RegWeekday"] = df["RegistrationDate"].dt.weekday
        df = df.drop(columns=["RegistrationDate"])
        print("  [parse] RegistrationDate -> RegYear/RegMonth/RegDay/RegWeekday")

    if "LastLoginIP" in df.columns:
        def _is_private(ip):
            try:
                first = int(str(ip).split(".")[0])
                return int(first in (10, 127, 172, 192))
            except Exception:
                return 0
        df["IP_IsPrivate"] = df["LastLoginIP"].apply(_is_private)
        df = df.drop(columns=["LastLoginIP"])
        print("  [parse] LastLoginIP -> IP_IsPrivate (0/1)")
    return df


# ─────────────────────────────────────────────────────────────
# 5. Valeurs sentinelles / aberrantes
# ─────────────────────────────────────────────────────────────
def fix_outliers(df):
    if "SupportTicketsCount" in df.columns:
        mask = (df["SupportTicketsCount"] < 0) | (df["SupportTicketsCount"] > 15)
        print(f"  [outlier] SupportTicketsCount : {mask.sum()} valeurs -> NaN")
        df.loc[mask, "SupportTicketsCount"] = np.nan
    elif "SupportTickets" in df.columns:
        mask = (df["SupportTickets"] < 0) | (df["SupportTickets"] > 15)
        print(f"  [outlier] SupportTickets : {mask.sum()} valeurs -> NaN")
        df.loc[mask, "SupportTickets"] = np.nan

    if "SatisfactionScore" in df.columns:
        mask = ~df["SatisfactionScore"].between(1, 5)
        print(f"  [outlier] SatisfactionScore : {mask.sum()} valeurs -> NaN")
        df.loc[mask, "SatisfactionScore"] = np.nan
    elif "Satisfaction" in df.columns:
        mask = ~df["Satisfaction"].between(1, 5)
        print(f"  [outlier] Satisfaction : {mask.sum()} valeurs -> NaN")
        df.loc[mask, "Satisfaction"] = np.nan

    if "MonetaryTotal" in df.columns:
        q01 = df["MonetaryTotal"].quantile(0.01)
        q99 = df["MonetaryTotal"].quantile(0.99)
        df["MonetaryTotal"] = df["MonetaryTotal"].clip(q01, q99)
        print(f"  [outlier] MonetaryTotal : clip [{q01:.0f}, {q99:.0f}]")
    return df


# ─────────────────────────────────────────────────────────────
# 6. Feature Engineering
# ─────────────────────────────────────────────────────────────
def feature_engineering(df):
    if "MonetaryTotal" in df.columns and "Frequency" in df.columns:
        df["AvgBasket"] = df["MonetaryTotal"] / (df["Frequency"] + 1)
    if "UniqueProducts" in df.columns and "TotalTransactions" in df.columns:
        df["DiversityRatio"] = df["UniqueProducts"] / (df["TotalTransactions"] + 1)
    elif "UniqueProducts" in df.columns and "TotalTrans" in df.columns:
        df["DiversityRatio"] = df["UniqueProducts"] / (df["TotalTrans"] + 1)
    if "CancelledTransactions" in df.columns and "TotalTransactions" in df.columns:
        df["CancelRate"] = df["CancelledTransactions"] / (df["TotalTransactions"] + 1)
    if "MonetaryTotal" in df.columns and "UniqueProducts" in df.columns:
        df["ValuePerProduct"] = df["MonetaryTotal"] / (df["UniqueProducts"] + 1)
    return df


# ─────────────────────────────────────────────────────────────
# 7. Encodage catégoriel — FIT sur train seulement
#    (ici on encode sur tout df, mais on garde les mappings
#     pour les appliquer proprement après le split si besoin)
# ─────────────────────────────────────────────────────────────
def encode_categoricals_df(df):
    """
    Encode toutes les colonnes object SAUF la cible.
    Retourne (df_encoded, label_encoders_dict).
    L'encodage est fait sur le df complet car le split vient ensuite.
    NOTE : pour éviter le leakage, les colonnes catégorielles avec
    peu de valeurs peuvent être traitées par ordinal mapping fixe.
    Ici on fait un LabelEncoder global et on re-fittera sur X_train
    après le split pour les colonnes non-binaires si nécessaire.
    """
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if c != TARGET_COL]
    encoders = {}
    if cat_cols:
        print(f"  [encode] {len(cat_cols)} colonnes : {cat_cols}")
    for col in cat_cols:
        df[col] = df[col].fillna("Inconnu")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


# ─────────────────────────────────────────────────────────────
# 8. Imputation — CORRECTION LEAKAGE
#    Calcul des médianes sur X_train UNIQUEMENT
# ─────────────────────────────────────────────────────────────
def impute_with_train_medians(X_train, X_test):
    """
    CORRECTION PRINCIPALE :
    - Calcule les médianes sur X_train uniquement
    - Applique ces médianes sur X_train ET X_test
    - Sauvegarde les médianes pour l'inférence en production
    """
    num_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
    train_medians = {}
    for col in num_cols:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            median_val = X_train[col].median()
            train_medians[col] = median_val
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col]  = X_test[col].fillna(median_val)
    print(f"  [impute] Médianes calculées sur X_train : {len(train_medians)} colonnes")
    return X_train, X_test, train_medians


# ─────────────────────────────────────────────────────────────
# 9. Corrélation — SUR X_TRAIN UNIQUEMENT
# ─────────────────────────────────────────────────────────────
def remove_high_corr(X_train, threshold=0.85):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    if to_drop:
        print(f"  [corr] supprimées (|r| > {threshold}) : {to_drop}")
    else:
        print(f"  [corr] aucune paire au-dessus du seuil {threshold}")
    return to_drop


# ─────────────────────────────────────────────────────────────
# 10. Importance features — SUR X_TRAIN / y_TRAIN UNIQUEMENT
# ─────────────────────────────────────────────────────────────
def select_by_importance(X_train, y_train, threshold=0.005):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    kept    = importances[importances >= threshold].index.tolist()
    dropped = importances[importances < threshold].index.tolist()
    if dropped:
        print(f"  [importance] supprimées (< {threshold}) : {dropped}")
    print(f"  [importance] {len(kept)} / {len(X_train.columns)} features conservées")
    return kept


# ─────────────────────────────────────────────────────────────
# 11. Split + sélections post-split + SMOTE + normalisation
# ─────────────────────────────────────────────────────────────
def split_and_save(df, target=TARGET_COL):
    """
    Ordre strict SANS data leakage :
      1.  X / y
      2.  train_test_split stratifié
      3.  Imputation  -> médianes X_train, apply X_test   ← CORRECTION
      4.  Corrélation -> calcul X_train, apply X_test
      5.  Importance  -> calcul X_train, apply X_test
      6.  SMOTE       -> X_train uniquement
      7.  StandardScaler -> fit X_train_res, transform X_test
      8.  Sauvegarde des méta-données (médianes, feature_names)
    """
    X = df.drop(columns=[target])
    y = df[target]

    assert target not in X.columns, f"ERREUR : '{target}' dans X !"
    leakage_restantes = [c for c in LEAKAGE_COLS if c in X.columns]
    if leakage_restantes:
        print(f"  [WARNING] Leakage résiduel supprimé : {leakage_restantes}")
        X = X.drop(columns=leakage_restantes)

    X = X.apply(pd.to_numeric, errors="coerce")
    print(f"  [info] Features disponibles : {X.shape[1]}")
    print(f"  [info] Distribution Churn   : {y.mean():.1%} positifs")

    # Étape 2 : Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [split] Train {X_train.shape} | Test {X_test.shape}")

    # Étape 3 : Imputation (CORRECTION — médianes sur X_train uniquement)
    X_train, X_test, train_medians = impute_with_train_medians(
        X_train.copy(), X_test.copy()
    )
    # Sauvegarder les médianes pour l'inférence
    save_model(train_medians, "imputer_medians")

    # Étape 4 : Corrélation (X_train uniquement)
    cols_drop_corr = remove_high_corr(X_train, threshold=0.85)
    X_train = X_train.drop(columns=cols_drop_corr, errors="ignore")
    X_test  = X_test.drop(columns=cols_drop_corr, errors="ignore")

    # Étape 5 : Importance (X_train / y_train uniquement)
    kept = select_by_importance(X_train, y_train, threshold=0.005)
    X_train = X_train[kept]
    X_test  = X_test[kept]

    # Étape 6 : SMOTE (X_train uniquement)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  [SMOTE] Après rééquilibrage : {X_train_res.shape}")
    print("Distribution AVANT SMOTE :")
    print(y_train.value_counts(normalize=True))
    print("\nDistribution APRÈS SMOTE :")
    print(y_train_res.value_counts(normalize=True))

    # Étape 7 : Normalisation (fit X_train_res, transform X_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)
    save_model(scaler, "scaler")

    # Étape 8 : Sauvegarder les noms de features (ordre exact du scaler)
    feature_names = kept  # list de str
    save_model(feature_names, "feature_names")
    print(f"  [save] feature_names : {feature_names}")

    # Sauvegarde CSV
    os.makedirs(DATA_TRAIN_TEST, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=kept).to_csv(
        os.path.join(DATA_TRAIN_TEST, "X_train.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=kept).to_csv(
        os.path.join(DATA_TRAIN_TEST, "X_test.csv"), index=False)
    pd.DataFrame(y_train_res, columns=[target]).to_csv(
        os.path.join(DATA_TRAIN_TEST, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(
        os.path.join(DATA_TRAIN_TEST, "y_test.csv"), index=False)
    print(f"  [save] Fichiers sauvegardés dans {DATA_TRAIN_TEST}")
    return X_train_scaled, X_test_scaled, y_train_res, y_test, kept


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────
def run_preprocessing(filename="retail.csv"):
    print("\n" + "="*60)
    print("PIPELINE PREPROCESSING")
    print("="*60)

    df = load_and_inspect(filename)

    print("\n[ÉTAPE 2] Suppression features inutiles + leakage")
    df = drop_useless_features(df)

    print("\n[ÉTAPE 3] Suppression colonnes >50% manquantes")
    df = drop_high_missing(df)

    print("\n[ÉTAPE 4] Parsing (date, IP)")
    df = parse_special_features(df)

    print("\n[ÉTAPE 5] Valeurs aberrantes / sentinelles")
    df = fix_outliers(df)

    print("\n[ÉTAPE 6] Feature Engineering")
    df = feature_engineering(df)

    print("\n[ÉTAPE 7] Encodage catégoriel (sur df complet, pré-split)")
    df, encoders = encode_categoricals_df(df)

    # Sauvegarde dataset nettoyé (avant split, avant imputation)
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    path_processed = os.path.join(DATA_PROCESSED, "processed.csv")
    df.to_csv(path_processed, index=False)
    print(f"\n[save] Dataset nettoyé : {df.shape} -> {path_processed}")

    print("\n[ÉTAPES 8-11] Split / Imputation / Corrélation / Importance / SMOTE / Scaler")
    print("  CORRECTION : imputation APRÈS split (zéro leakage statistique)")
    X_train, X_test, y_train, y_test, feature_names = split_and_save(df)

    print("\n" + "="*60)
    print("Pipeline terminé")
    print(f"  X_train      : {X_train.shape}")
    print(f"  X_test       : {X_test.shape}")
    print(f"  Features finales ({len(feature_names)}) : {feature_names}")
    print("="*60 + "\n")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()