# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
train_model.py — Entraînement des modèles ML
Projet ML Retail — Analyse Comportementale Clientèle

Modèles :
1. ACP             — Réduction dimensionnelle + visualisation
2. Clustering      — K-Means (segmentation clients)
3. Classification  — Random Forest + Gradient Boosting (prédiction Churn)
4. Régression      — Random Forest Regressor (prédiction MonetaryTotal)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier)
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              mean_absolute_error, mean_squared_error, r2_score,
                              silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (save_model, plot_feature_importance,
                   REPORTS_DIR, DATA_TRAIN_TEST, MODELS_DIR)

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _encode_and_clean(df):
    """
    Encode toutes les colonnes texte résiduelles (relues comme object depuis CSV)
    puis force tout en numérique. Utilisé partout avant fit().
    """
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if str_cols:
        print(f"  ⚙️  Encodage auto des colonnes texte résiduelles : {str_cols}")
        for col in str_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def _encode_pair(X_train, X_test):
    """Encode les colonnes texte sur train puis applique sur test (même mapping)."""
    str_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    if str_cols:
        print(f"  ⚙️  Encodage auto des colonnes texte résiduelles : {str_cols}")
        for col in str_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            mapping = {v: i for i, v in enumerate(le.classes_)}
            X_test[col] = X_test[col].astype(str).map(mapping).fillna(0).astype(int)
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X_train, X_test


# ── Chargement ───────────────────────────────────────────────────────────────
def load_train_test():
    X_train = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "y_test.csv")).squeeze()
    X_train, X_test = _encode_pair(X_train, X_test)
    print(f"📂 Train : {X_train.shape} | Test : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 : ACP
# ════════════════════════════════════════════════════════════════════════════
def run_pca(X_train, X_test, n_components=0.95):
    """ACP : réduit la dimension en conservant 95% de la variance."""
    print("\n" + "="*60)
    print("MODULE 1 : ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("="*60)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    print(f"  ✅ Composantes retenues : {pca.n_components_}")
    print(f"  ✅ Variance expliquée   : {pca.explained_variance_ratio_.cumsum()[-1]:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.cumsum(pca.explained_variance_ratio_) * 100,
                 color="#3498db", linewidth=2, marker="o", markersize=3)
    axes[0].axhline(95, color="red", linestyle="--", label="95% seuil")
    axes[0].set_xlabel("Nombre de composantes")
    axes[0].set_ylabel("Variance expliquée cumulée (%)")
    axes[0].set_title("Courbe de variance expliquée — ACP")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train)
    axes[1].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.4, s=10, color="#9b59b6")
    axes[1].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("Projection 2D — ACP"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "pca_analysis.png"), dpi=120, bbox_inches="tight")
    plt.close()
    save_model(pca, "pca")
    return X_train_pca, X_test_pca, pca


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 : CLUSTERING K-MEANS
# ════════════════════════════════════════════════════════════════════════════
def run_kmeans(X_train_pca, X_train_original, k_range=range(2, 9)):
    """K-Means avec sélection du k optimal via Elbow + Silhouette."""
    print("\n" + "="*60)
    print("MODULE 2 : CLUSTERING K-MEANS")
    print("="*60)

    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_train_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_train_pca, labels,
                               sample_size=min(1000, len(X_train_pca)))
        silhouettes.append(sil)
        print(f"  k={k} | Inertie={km.inertia_:.0f} | Silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  🏆 Meilleur k (silhouette) : {best_k}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2)
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"k={best_k}")
    axes[0].set_xlabel("Nombre de clusters (k)"); axes[0].set_ylabel("Inertie (WCSS)")
    axes[0].set_title("Méthode Elbow"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "gs-", linewidth=2)
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"k={best_k}")
    axes[1].set_xlabel("Nombre de clusters (k)"); axes[1].set_ylabel("Score de Silhouette")
    axes[1].set_title("Score de Silhouette"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "kmeans_selection.png"), dpi=120, bbox_inches="tight")
    plt.close()

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_pca)

    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train_pca)
    palette = plt.cm.tab10(np.linspace(0, 1, best_k))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(best_k):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], color=palette[i],
                   label=f"Cluster {i} (n={mask.sum()})", alpha=0.6, s=15)
    ax.set_title(f"K-Means — {best_k} clusters (projection 2D)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "kmeans_clusters.png"), dpi=120, bbox_inches="tight")
    plt.close()

    cluster_df = pd.DataFrame(X_train_original).copy()
    cluster_df["Cluster"] = labels
    cluster_df.groupby("Cluster").mean().round(3).to_csv(
        os.path.join(REPORTS_DIR, "cluster_profiles.csv"))
    print(f"  ✅ Profils clusters sauvegardés")

    save_model(kmeans, "kmeans")
    return kmeans, labels, best_k



# ── Pipeline principal ───────────────────────────────────────────────────────
def run_all_models():
    X_train, X_test, y_train, y_test = load_train_test()

    # MODULE 1 — ACP
    X_train_pca, X_test_pca, pca = run_pca(X_train, X_test)

    # MODULE 2 — Clustering
    kmeans, labels, best_k = run_kmeans(X_train_pca, X_train)


    print("\n✅ Tous les modèles entraînés et sauvegardés !\n")


if __name__ == "__main__":
    run_all_models()