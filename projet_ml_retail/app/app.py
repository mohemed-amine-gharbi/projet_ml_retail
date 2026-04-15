# -*- coding: utf-8 -*-
"""
app.py — Interface Web Flask
Projet ML Retail — Analyse Comportementale Clientèle

Routes :
    GET  /          → Page d'accueil / formulaire client
    POST /predict   → Résultats de prédiction
    GET  /dashboard → Tableau de bord des résultats
    GET  /api/predict  → API REST (JSON)
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
import warnings
warnings.filterwarnings("ignore")

# Ajout du chemin src au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_TRAIN_TEST = os.path.join(BASE_DIR, "data", "train_test")


def load_models():
    """Charge les modèles ML."""
    import joblib
    models = {}
    for name in ["classifier_churn", "scaler", "pca", "kmeans", "regressor_monetary"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


MODELS = {}

@app.before_request
def load_on_first_request():
    global MODELS
    if not MODELS:
        MODELS = load_models()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération des features depuis le formulaire
        features = {
            "Recency": float(request.form.get("recency", 30)),
            "Frequency": float(request.form.get("frequency", 5)),
            "MonetaryTotal": float(request.form.get("monetary_total", 500)),
            "MonetaryAvg": float(request.form.get("monetary_avg", 100)),
            "CustomerTenure": float(request.form.get("customer_tenure", 180)),
            "ReturnRatio": float(request.form.get("return_ratio", 0.05)),
            "SupportTickets": float(request.form.get("support_tickets", 1)),
            "Satisfaction": float(request.form.get("satisfaction", 3.5)),
            "Age": float(request.form.get("age", 35)),
        }

        # Prédiction Churn
        result = {"error": None}

        if "classifier_churn" in MODELS:
            X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
            # On utilise les vraies features du modèle
            # Pour une demo, on prend une ligne et on remplace les valeurs clés
            sample = X_test.iloc[0:1].copy()
            for col in features:
                if col in sample.columns:
                    sample[col] = features[col]

            churn_proba = MODELS["classifier_churn"].predict_proba(sample)[0, 1]
            churn_pred = int(churn_proba >= 0.5)

            if churn_proba < 0.25:
                risk = "Faible"
                risk_color = "#27ae60"
                risk_icon = "[OK]"
            elif churn_proba < 0.50:
                risk = "Moyen"
                risk_color = "#f39c12"
                risk_icon = "[WARN]️"
            elif churn_proba < 0.75:
                risk = "Élevé"
                risk_color = "#e67e22"
                risk_icon = "🔶"
            else:
                risk = "Critique"
                risk_color = "#e74c3c"
                risk_icon = "🚨"

            result.update({
                "churn_proba": round(float(churn_proba) * 100, 1),
                "churn_pred": churn_pred,
                "risk": risk,
                "risk_color": risk_color,
                "risk_icon": risk_icon,
            })

        if "pca" in MODELS and "kmeans" in MODELS:
            X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
            sample = X_test.iloc[0:1].copy()
            X_pca = MODELS["pca"].transform(sample)
            segment = int(MODELS["kmeans"].predict(X_pca)[0])
            result["segment"] = segment

        result["features"] = features
        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html",
                               result={"error": str(e), "features": {}})


@app.route("/dashboard")
def dashboard():
    """Tableau de bord avec métriques globales."""
    stats = {}
    churn_path = os.path.join(REPORTS_DIR, "churn_predictions.csv")
    if os.path.exists(churn_path):
        df = pd.read_csv(churn_path)
        stats["total_clients"] = len(df)
        if "Churn_Predicted" in df.columns:
            stats["churned"] = int(df["Churn_Predicted"].sum())
            stats["churn_rate"] = round(df["Churn_Predicted"].mean() * 100, 1)
        if "Risk_Level" in df.columns:
            risk_counts = df["Risk_Level"].value_counts().to_dict()
            stats["risk_distribution"] = risk_counts

    segment_path = os.path.join(REPORTS_DIR, "segment_predictions.csv")
    if os.path.exists(segment_path):
        df_seg = pd.read_csv(segment_path)
        seg_counts = df_seg["Segment"].value_counts().sort_index().to_dict()
        stats["segment_distribution"] = {f"Cluster {k}": v for k, v in seg_counts.items()}

    return render_template("dashboard.html", stats=stats)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API REST pour prédictions depuis d'autres applications."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        if "classifier_churn" not in MODELS:
            return jsonify({"error": "Model not loaded"}), 500

        X_test = pd.read_csv(os.path.join(DATA_TRAIN_TEST, "X_test.csv"))
        sample = X_test.iloc[0:1].copy()
        for col, val in data.items():
            if col in sample.columns:
                sample[col] = float(val)

        churn_proba = float(MODELS["classifier_churn"].predict_proba(sample)[0, 1])

        if churn_proba < 0.25:
            risk = "Faible"
        elif churn_proba < 0.50:
            risk = "Moyen"
        elif churn_proba < 0.75:
            risk = "Élevé"
        else:
            risk = "Critique"

        segment = -1
        if "pca" in MODELS and "kmeans" in MODELS:
            X_pca = MODELS["pca"].transform(sample)
            segment = int(MODELS["kmeans"].predict(X_pca)[0])

        return jsonify({
            "churn_probability": round(churn_proba, 4),
            "churn_predicted": int(churn_proba >= 0.5),
            "risk_level": risk,
            "segment": segment
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[RUN] Démarrage de l'application Flask — Retail ML")
    print("   URL : http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
