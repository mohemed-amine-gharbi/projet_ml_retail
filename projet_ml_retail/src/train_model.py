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

# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 : CLASSIFICATION — CHURN
# ════════════════════════════════════════════════════════════════════════════
def run_classification(X_train, X_test, y_train, y_test):
    """Random Forest + Gradient Boosting pour prédire le Churn."""
    print("\n" + "="*60)
    print("MODULE 3 : CLASSIFICATION — PRÉDICTION CHURN")
    print("="*60)
    print(f"  Distribution Churn — Train : {y_train.mean():.1%} | Test : {y_test.mean():.1%}")

    # ─────────────────────────────────────────────
    # RANDOM FOREST
    # ─────────────────────────────────────────────
    print("\n  ⚙️  Random Forest en cours...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predictions
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # AUC TEST
    rf_auc = roc_auc_score(y_test, rf_proba)

    # 🔥 AUC TRAIN vs TEST
    rf_train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])

    print(f"  ✅ Random Forest — AUC Test : {rf_auc:.4f}")
    print(f"  📊 Random Forest — AUC Train: {rf_train_auc:.4f}")

    print(classification_report(y_test, rf_pred, target_names=["Fidèle", "Churned"]))

    # ─────────────────────────────────────────────
    # GRADIENT BOOSTING
    # ─────────────────────────────────────────────
    print("\n  ⚙️  Gradient Boosting en cours...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)

    # Predictions
    gb_pred  = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]

    # AUC TEST
    gb_auc = roc_auc_score(y_test, gb_proba)

    # 🔥 AUC TRAIN vs TEST
    gb_train_auc = roc_auc_score(y_train, gb.predict_proba(X_train)[:, 1])

    print(f"  ✅ Gradient Boosting — AUC Test : {gb_auc:.4f}")
    print(f"  📊 Gradient Boosting — AUC Train: {gb_train_auc:.4f}")

    print(classification_report(y_test, gb_pred, target_names=["Fidèle", "Churned"]))

    # ─────────────────────────────────────────────
    # BEST MODEL
    # ─────────────────────────────────────────────
    best_clf  = rf if rf_auc >= gb_auc else gb
    best_name = "RandomForest" if rf_auc >= gb_auc else "GradientBoosting"
    best_pred = rf_pred if rf_auc >= gb_auc else gb_pred
    best_proba= rf_proba if rf_auc >= gb_auc else gb_proba

    print(f"\n  🏆 Meilleur modèle : {best_name} (AUC={max(rf_auc, gb_auc):.4f})")

    # ─────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_proba)

    axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})", lw=2)
    axes[0].plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC={gb_auc:.3f})", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)

    axes[0].set_xlabel("Taux de faux positifs")
    axes[0].set_ylabel("Taux de vrais positifs")
    axes[0].set_title("Courbe ROC — Prédiction Churn")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Fidèle", "Churned"],
                yticklabels=["Fidèle", "Churned"])

    axes[1].set_title(f"Matrice de confusion — {best_name}")
    axes[1].set_xlabel("Prédit")
    axes[1].set_ylabel("Réel")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "classification_results.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────
    if hasattr(best_clf, "feature_importances_"):
        plot_feature_importance(
            best_clf,
            X_train.columns,
            title=f"Importance des features — {best_name}"
        )

    # ─────────────────────────────────────────────
    # SAVE MODELS
    # ─────────────────────────────────────────────
    save_model(best_clf, "classifier_churn")
    save_model(rf, "random_forest_churn")
    save_model(gb, "gradient_boosting_churn")

    return best_clf

# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 : RÉGRESSION — MonetaryTotal
# ════════════════════════════════════════════════════════════════════════════
def run_regression(X_train, X_test, y_train, y_test):
    """Random Forest Regressor pour prédire MonetaryTotal."""
    print("\n" + "="*60)
    print("MODULE 4 : RÉGRESSION — PRÉDICTION MonetaryTotal")
    print("="*60)

    # ── CORRECTION BUG : encoder les colonnes texte résiduelles ────────────
    X_train = _encode_and_clean(X_train.copy())
    X_test  = _encode_and_clean(X_test.copy())

    rfr = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"  ✅ MAE  : {mae:.2f} £")
    print(f"  ✅ RMSE : {rmse:.2f} £")
    print(f"  ✅ R²   : {r2:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lim = max(abs(y_test.min()), abs(y_test.max()), abs(y_pred.min()), abs(y_pred.max()))
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=10, color="#27ae60")
    axes[0].plot([-lim, lim], [-lim, lim], "r--", lw=1.5)
    axes[0].set_xlabel("Réel (£)"); axes[0].set_ylabel("Prédit (£)")
    axes[0].set_title(f"Réel vs Prédit — R²={r2:.3f}"); axes[0].grid(alpha=0.3)

    residuals = y_test.values - y_pred
    axes[1].hist(residuals, bins=40, color="#f39c12", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Résidus (£)"); axes[1].set_ylabel("Fréquence")
    axes[1].set_title(f"Distribution des résidus — MAE={mae:.1f}£"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "regression_results.png"), dpi=120, bbox_inches="tight")
    plt.close()

    save_model(rfr, "regressor_monetary")
    return rfr


# ── Pipeline principal ───────────────────────────────────────────────────────
def run_all_models():
    X_train, X_test, y_train, y_test = load_train_test()

    # MODULE 1 — ACP
    X_train_pca, X_test_pca, pca = run_pca(X_train, X_test)

    # MODULE 2 — Clustering
    kmeans, labels, best_k = run_kmeans(X_train_pca, X_train)

    # MODULE 3 — Classification Churn
    best_clf = run_classification(X_train, X_test, y_train, y_test)

    # MODULE 4 — Régression MonetaryTotal
    processed_path = os.path.join(
        os.path.dirname(DATA_TRAIN_TEST), "processed", "processed.csv"
    )
    if os.path.exists(processed_path):
        df_processed = pd.read_csv(processed_path)
        # Chercher la colonne MonetaryTotal (nom exact ou variante)
        monetary_col = None
        for candidate in ["MonetaryTotal", "Monetary", "monetary_total"]:
            if candidate in df_processed.columns:
                monetary_col = candidate
                break

        if monetary_col:
            X_reg = df_processed.drop(columns=[monetary_col, "Churn"], errors="ignore")
            y_reg = df_processed[monetary_col]
            # ── Encoder les colonnes texte ICI avant le split ──────────────
            X_reg = _encode_and_clean(X_reg)
            Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                X_reg, y_reg, test_size=0.2, random_state=42
            )
            sc = StandardScaler()
            num_cols = Xr_train.select_dtypes(include=[np.number]).columns
            Xr_train[num_cols] = sc.fit_transform(Xr_train[num_cols])
            Xr_test[num_cols]  = sc.transform(Xr_test[num_cols])
            run_regression(Xr_train, Xr_test, yr_train, yr_test)
        else:
            print("\n⚠️  MonetaryTotal introuvable dans processed — régression ignorée")
    else:
        print("\n⚠️  Données processed non trouvées — exécutez preprocessing.py d'abord")

    print("\n✅ Tous les modèles entraînés et sauvegardés !\n")


if __name__ == "__main__":
    run_all_models()