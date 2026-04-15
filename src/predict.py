# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
predict.py — Prédictions sur de nouvelles données
Projet ML Retail — Analyse Comportementale Clientèle

Usage :
    python predict.py --mode churn      # Prédiction churn sur X_test
    python predict.py --mode segment    # Segmentation K-Means
    python predict.py --mode monetary   # Prédiction MonetaryTotal
    python predict.py --mode all        # Tout (défaut)
"""

import numpy as np
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, DATA_TRAIN_TEST, REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Helper : encode les colonnes texte résiduelles ───────────────────────────
def _encode_and_clean(df):
    """Encode les colonnes object et force tout en numérique."""
    from sklearn.preprocessing import LabelEncoder
    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


# ── Helper : aligne les colonnes de X sur celles attendues par le modèle ─────
def _align_features(X, model):
    """
    Ajoute les colonnes manquantes (valeur 0) et retire les colonnes
    inconnues du modèle, pour éviter le ValueError sur feature_names.
    """
    if not hasattr(model, "feature_names_in_"):
        return X  # modèle sans contrainte de noms
    expected = list(model.feature_names_in_)
    # Ajouter les colonnes manquantes
    for col in expected:
        if col not in X.columns:
            X[col] = 0
    # Conserver uniquement les colonnes attendues, dans le bon ordre
    return X[expected]


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION CHURN
# ════════════════════════════════════════════════════════════════════════════
def predict_churn(X_new=None):
    """Prédit la probabilité de churn pour chaque client."""
    print("\n🔮 Prédiction CHURN")
    print("="*50)

    clf = load_model("classifier_churn")

    if X_new is None:
        X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "y_test.csv")).squeeze()
    else:
        X_test = X_new.copy()
        y_test = None

    X_test = _encode_and_clean(X_test)
    X_test = _align_features(X_test, clf)

    predictions  = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "Index":            range(len(predictions)),
        "Churn_Predicted":  predictions,
        "Churn_Probability": np.round(probabilities, 4),
        "Risk_Level": pd.cut(
            probabilities,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["Faible", "Moyen", "Élevé", "Critique"]
        )
    })

    if y_test is not None:
        results["Churn_Actual"] = y_test.values
        results["Correct"] = results["Churn_Predicted"] == results["Churn_Actual"]
        print(f"  ✅ Accuracy : {results['Correct'].mean():.2%}")

    high_risk = results[results["Risk_Level"].isin(["Élevé", "Critique"])]
    print(f"  ⚠️  Clients à risque élevé/critique : {len(high_risk)} ({len(high_risk)/len(results):.1%})")
    print(f"\n  📊 Distribution des niveaux de risque :")
    print(results["Risk_Level"].value_counts().to_string())

    output_path = os.path.join(REPORTS_DIR, "churn_predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\n  ✅ Prédictions sauvegardées : {output_path}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION SEGMENT K-MEANS
# ════════════════════════════════════════════════════════════════════════════
def predict_segment(X_new=None):
    """Assigne chaque client à un segment K-Means."""
    print("\n🔮 Prédiction SEGMENT (K-Means)")
    print("="*50)

    kmeans = load_model("kmeans")
    pca    = load_model("pca")

    if X_new is None:
        X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
    else:
        X_test = X_new.copy()

    X_test = _encode_and_clean(X_test)
    # L'ACP attend le même nombre de features qu'à l'entraînement
    if hasattr(pca, "n_features_in_"):
        n_expected = pca.n_features_in_
        # Ajouter/retirer des colonnes si nécessaire
        current_cols = X_test.shape[1]
        if current_cols < n_expected:
            for i in range(n_expected - current_cols):
                X_test[f"_pad_{i}"] = 0
        elif current_cols > n_expected:
            X_test = X_test.iloc[:, :n_expected]

    X_pca    = pca.transform(X_test)
    segments = kmeans.predict(X_pca)

    results = pd.DataFrame({
        "Index":   range(len(segments)),
        "Segment": segments,
    })
    print(f"  📊 Distribution des segments :")
    print(results["Segment"].value_counts().sort_index().to_string())

    output_path = os.path.join(REPORTS_DIR, "segment_predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\n  ✅ Segments sauvegardés : {output_path}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION MONETARY TOTAL
# ════════════════════════════════════════════════════════════════════════════
def predict_monetary(X_new=None):
    """Prédit le MonetaryTotal pour de nouveaux clients."""
    print("\n🔮 Prédiction MONETARY TOTAL")
    print("="*50)

    try:
        rfr = load_model("regressor_monetary")
    except Exception:
        print("  ⚠️  Modèle de régression non trouvé. Exécutez train_model.py d'abord.")
        return None

    if X_new is None:
        # Le régresseur a été entraîné sur projet_processed SANS MonetaryTotal
        # → on charge X_test mais on retire MonetaryTotal s'il est présent
        X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
    else:
        X_test = X_new.copy()

    # ── CORRECTION BUG : retirer MonetaryTotal et les colonnes inconnues ───
    cols_to_drop = [c for c in ["MonetaryTotal", "Churn"] if c in X_test.columns]
    if cols_to_drop:
        X_test = X_test.drop(columns=cols_to_drop)

    X_test = _encode_and_clean(X_test)
    X_test = _align_features(X_test, rfr)  # aligne sur les features du fit

    predictions = rfr.predict(X_test)

    results = pd.DataFrame({
        "Index": range(len(predictions)),
        "MonetaryTotal_Predicted": np.round(predictions, 2)
    })
    print(f"  ✅ Dépense prédite moyenne : {predictions.mean():.2f} £")
    print(f"  ✅ Dépense prédite médiane : {np.median(predictions):.2f} £")
    print(f"  ✅ Min / Max : {predictions.min():.2f} £ / {predictions.max():.2f} £")

    output_path = os.path.join(REPORTS_DIR, "monetary_predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\n  ✅ Prédictions sauvegardées : {output_path}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION CLIENT UNIQUE (utilisé par Flask)
# ════════════════════════════════════════════════════════════════════════════
def predict_single_client(features_dict):
    """
    Prédit pour un seul client (utilisé par l'API Flask).
    features_dict : dictionnaire {feature_name: value}
    Retourne : {'churn_probability': float, 'risk_level': str, 'segment': int}
    """
    X = pd.DataFrame([features_dict])
    X = _encode_and_clean(X)

    # Churn
    churn_proba, risk = None, "Inconnu"
    try:
        clf = load_model("classifier_churn")
        X_clf = _align_features(X.copy(), clf)
        churn_proba = clf.predict_proba(X_clf)[0, 1]
        risk = ("Faible"   if churn_proba < 0.25 else
                "Moyen"    if churn_proba < 0.50 else
                "Élevé"    if churn_proba < 0.75 else
                "Critique")
    except Exception as e:
        print(f"  ⚠️  Churn prediction error : {e}")

    # Segment
    segment = -1
    try:
        pca    = load_model("pca")
        kmeans = load_model("kmeans")
        X_pca  = pca.transform(X)
        segment = int(kmeans.predict(X_pca)[0])
    except Exception as e:
        print(f"  ⚠️  Segment prediction error : {e}")

    return {
        "churn_probability": round(float(churn_proba), 4) if churn_proba is not None else None,
        "risk_level":  risk,
        "segment":     segment
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédictions ML Retail")
    parser.add_argument("--mode",
                        choices=["churn", "segment", "monetary", "all"],
                        default="all",
                        help="Type de prédiction à effectuer")
    args = parser.parse_args()

    if args.mode in ("churn", "all"):
        predict_churn()
    if args.mode in ("segment", "all"):
        predict_segment()
    if args.mode in ("monetary", "all"):
        predict_monetary()