# -*- coding: utf-8 -*-
import sys
import io
# Fix Windows encoding (cp1252 -> utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
utils.py — Fonctions utilitaires partagées
Projet ML Retail — Analyse Comportementale Clientèle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── Chemins du projet ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_TRAIN_TEST = os.path.join(BASE_DIR, "data", "train_test")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_raw_data(filename="retail.csv"):
    """Charge les données brutes."""
    path = os.path.join(DATA_RAW, filename)
    df = pd.read_csv(path, low_memory=False)
    print(f"[DIR] Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def summarize_dataframe(df):
    """Résumé rapide de la qualité des données."""
    print("\n" + "="*60)
    print("RÉSUMÉ DU DATASET")
    print("="*60)
    print(f"Dimensions       : {df.shape}")
    print(f"Types            : {df.dtypes.value_counts().to_dict()}")
    print(f"\nValeurs manquantes (top 10):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, cnt in missing.head(10).items():
            print(f"  {col:30s}: {cnt:5d} ({cnt/len(df):.1%})")
    else:
        print("  Aucune valeur manquante")
    print(f"\nDoublons         : {df.duplicated().sum()}")
    print("="*60)


def plot_correlation_heatmap(df, cols=None, figsize=(16, 12), title="Matrice de Corrélation"):
    """Heatmap de corrélation avec seuil visuel."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, linewidths=0.3, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap sauvegardée : {path}")
    return corr


def find_high_correlation_pairs(df, threshold=0.8):
    """Retourne les paires de features fortement corrélées."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] >= threshold:
                pairs.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": round(corr.iloc[i, j], 4)
                })
    pairs_df = pd.DataFrame(pairs).sort_values("correlation", ascending=False)
    print(f"\n[>>] Paires corrélées (|r| ≥ {threshold}) : {len(pairs_df)}")
    print(pairs_df.to_string(index=False))
    return pairs_df


def plot_class_distribution(y, title="Distribution des classes", save_name="class_distribution.png"):
    """Visualise la distribution d'une variable cible."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=["#2ecc71", "#e74c3c"][:len(counts)])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'observations")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val}\n({val/len(y):.1%})", ha="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, save_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Distribution sauvegardée : {path}")


def save_model(model, name):
    """Sauvegarde un modèle sklearn."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[OK] Modèle sauvegardé : {path}")
    return path


def load_model(name):
    """Charge un modèle sauvegardé."""
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    model = joblib.load(path)
    print(f"[PKG] Modèle chargé : {path}")
    return model


def plot_feature_importance(model, feature_names, top_n=20, title="Importance des features"):
    """Visualise l'importance des features pour les modèles arborescents."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    importances.plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Gini)")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Feature importance sauvegardée : {path}")
