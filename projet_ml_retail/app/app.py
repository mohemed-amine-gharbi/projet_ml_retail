# -*- coding: utf-8 -*-
import os, sys, traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

app = Flask(__name__)

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
DATA_TRAIN_TEST = os.path.join(BASE_DIR, "data", "train_test")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")

# ════════════════════════════════════════════════════════════════════════════
# ÉTAT GLOBAL
# ════════════════════════════════════════════════════════════════════════════
MODELS         = {}
FEATURE_NAMES  = []   # liste ordonnée des features attendues par scaler/classifier
TRAIN_MEDIANS  = {}   # médianes de X_train pour imputation (zéro leakage)
REGRESSOR_COLS = []   # features du régresseur

def _load_all():
    global MODELS, FEATURE_NAMES, TRAIN_MEDIANS, REGRESSOR_COLS
    import joblib

    for name in ["classifier_churn", "scaler", "pca", "kmeans",
                 "regressor_monetary", "feature_names", "imputer_medians"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            MODELS[name] = joblib.load(path)
            print(f"[OK] {name}")
        else:
            print(f"[WARN] absent : {name}.joblib")

    # ── Récupérer la liste exacte des features (ordre scaler) ────────────
    if "feature_names" in MODELS:
        FEATURE_NAMES = list(MODELS["feature_names"])
        print(f"[OK] FEATURE_NAMES depuis feature_names.joblib : {FEATURE_NAMES}")
    elif "scaler" in MODELS and hasattr(MODELS["scaler"], "feature_names_in_"):
        FEATURE_NAMES = list(MODELS["scaler"].feature_names_in_)
        print(f"[OK] FEATURE_NAMES depuis scaler.feature_names_in_")
    else:
        path = os.path.join(DATA_TRAIN_TEST, "X_test.csv")
        if os.path.exists(path):
            FEATURE_NAMES = pd.read_csv(path, nrows=0).columns.tolist()
            print(f"[OK] FEATURE_NAMES depuis X_test.csv")

    # ── Médianes d'imputation ─────────────────────────────────────────────
    if "imputer_medians" in MODELS:
        TRAIN_MEDIANS = dict(MODELS["imputer_medians"])
        print(f"[OK] TRAIN_MEDIANS : {len(TRAIN_MEDIANS)} colonnes")

    # ── Colonnes du régresseur ────────────────────────────────────────────
    if "regressor_monetary" in MODELS and hasattr(MODELS["regressor_monetary"], "feature_names_in_"):
        REGRESSOR_COLS = list(MODELS["regressor_monetary"].feature_names_in_)

    print(f"[OK] FEATURE_NAMES ({len(FEATURE_NAMES)}) : {FEATURE_NAMES}")
    print(f"[OK] REGRESSOR_COLS ({len(REGRESSOR_COLS)})")


@app.before_request
def init():
    if not MODELS:
        _load_all()


# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU VECTEUR DE FEATURES POUR LE PIPELINE CHURN/SEGMENT
# ════════════════════════════════════════════════════════════════════════════
def _build_feature_vector(u):
    """
    Construit le vecteur de features BRUTES (non normalisées) dans l'ordre
    exact de FEATURE_NAMES, à partir des inputs utilisateur.

    Logique :
      1. Toutes les features connues directement depuis le formulaire
      2. Features dérivées calculées à partir des inputs
      3. Valeurs par défaut (médiane d'entraînement ou 0) pour le reste
      4. Imputation avec les médianes de X_train (pas de leakage)
    """
    freq         = float(u.get("Frequency", 5))
    mon_total    = float(u.get("MonetaryTotal", 500))
    mon_avg      = float(u.get("MonetaryAvg", 100))
    tenure       = float(u.get("CustomerTenureDays", 180))
    ret_ratio    = float(u.get("ReturnRatio", 0.05))
    support_tick = float(u.get("SupportTicketsCount", 1))
    satisfaction = float(u.get("SatisfactionScore", 3.5))
    age          = float(u.get("Age", 35))
    reg_year     = float(u.get("RegYear", 2011))
    reg_month    = float(u.get("RegMonth", 6))

    # Features dérivées calculables depuis les inputs
    derived = {
        "Frequency":                 freq,
        "MonetaryTotal":             mon_total,
        "MonetaryAvg":               mon_avg,
        "MonetaryStd":               mon_avg * 0.30,
        "MonetaryMin":               max(mon_avg * 0.2, 1.0),
        "MonetaryMax":               mon_avg * 3.0,
        "TotalQuantity":             freq * 2.0,
        "AvgQuantityPerTransaction": 2.0,
        "MinQuantity":               1.0,
        "MaxQuantity":               5.0,
        "CustomerTenureDays":        tenure,
        "FirstPurchaseDaysAgo":      tenure + 30.0,
        "PreferredDayOfWeek":        2.0,
        "PreferredHour":             14.0,
        "PreferredMonth":            reg_month,
        "WeekendPurchaseRatio":      0.28,
        "AvgDaysBetweenPurchases":   tenure / max(freq, 1),
        "UniqueProducts":            max(freq * 1.5, 1.0),
        "UniqueDescriptions":        max(freq * 1.6, 1.0),
        "AvgProductsPerTransaction": 1.5,
        "UniqueCountries":           1.0,
        "NegativeQuantityCount":     ret_ratio * freq,
        "ZeroPriceCount":            0.0,
        "CancelledTransactions":     ret_ratio * freq * 0.5,
        "ReturnRatio":               ret_ratio,
        "TotalTransactions":         freq,
        "UniqueInvoices":            freq,
        "AvgLinesPerInvoice":        2.0,
        "Age":                       age,
        "SupportTicketsCount":       support_tick,
        "SatisfactionScore":         satisfaction,
        "AgeCategory":               (0.0 if age < 30 else (1.0 if age < 50 else 2.0)),
        "SpendingCategory":          (0.0 if mon_total < 300 else (1.0 if mon_total < 1000 else 2.0)),
        "FavoriteSeason":            1.5,
        "PreferredTimeOfDay":        1.0,
        "Region":                    0.0,
        "WeekendPreference":         0.0,
        "BasketSizeCategory":        (0.0 if mon_avg < 50 else (1.0 if mon_avg < 200 else 2.0)),
        "ProductDiversity":          max(freq * 1.5, 1.0) / max(freq * 2.0, 1),
        "Gender":                    0.0,
        "Country":                   0.0,
        "RegYear":                   reg_year,
        "RegMonth":                  reg_month,
        "RegDay":                    15.0,
        "RegWeekday":                2.0,
        "IP_IsPrivate":              1.0,
        "AvgBasket":                 mon_total / max(freq, 1),
        "DiversityRatio":            max(freq * 1.5, 1.0) / max(freq + 1, 1),
        "CancelRate":                (ret_ratio * freq * 0.5) / max(freq + 1, 1),
        "ValuePerProduct":           mon_total / max(max(freq * 1.5, 1.0) + 1, 1),
    }

    if not FEATURE_NAMES:
        # Fallback si les noms ne sont pas chargés
        return pd.DataFrame([derived])

    # Construire le vecteur dans l'ordre exact des FEATURE_NAMES
    row = {}
    for col in FEATURE_NAMES:
        if col in derived:
            row[col] = derived[col]
        elif col in TRAIN_MEDIANS:
            # Utiliser la médiane de X_train (zéro leakage)
            row[col] = TRAIN_MEDIANS[col]
        else:
            row[col] = 0.0

    df = pd.DataFrame([row])[FEATURE_NAMES]
    print(f"[FEATURE_VECTOR] shape={df.shape}, cols={list(df.columns)}")
    return df


def _build_feature_vector_reg(u):
    """Vecteur de features pour le régresseur MonetaryTotal."""
    freq         = float(u.get("Frequency", 5))
    mon_avg      = float(u.get("MonetaryAvg", 100))
    tenure       = float(u.get("CustomerTenureDays", 180))
    ret_ratio    = float(u.get("ReturnRatio", 0.05))
    age          = float(u.get("Age", 35))
    satisfaction = float(u.get("SatisfactionScore", 3.5))
    support_tick = float(u.get("SupportTicketsCount", 1))
    reg_year     = float(u.get("RegYear", 2011))
    reg_month    = float(u.get("RegMonth", 6))

    raw_map = {
        "Frequency": freq, "MonetaryAvg": mon_avg,
        "MonetaryStd": mon_avg * 0.30,
        "MonetaryMin": max(mon_avg * 0.2, 1.0),
        "MonetaryMax": mon_avg * 3.0,
        "TotalQuantity": freq * 2.0,
        "AvgQuantityPerTransaction": 2.0, "MinQuantity": 1.0, "MaxQuantity": 5.0,
        "CustomerTenureDays": tenure, "FirstPurchaseDaysAgo": tenure + 30.0,
        "PreferredDayOfWeek": 2.0, "PreferredHour": 14.0,
        "PreferredMonth": reg_month, "WeekendPurchaseRatio": 0.28,
        "AvgDaysBetweenPurchases": tenure / max(freq, 1),
        "UniqueProducts": max(freq * 1.5, 1.0),
        "UniqueDescriptions": max(freq * 1.6, 1.0),
        "AvgProductsPerTransaction": 1.5, "UniqueCountries": 1.0,
        "NegativeQuantityCount": ret_ratio * freq, "ZeroPriceCount": 0.0,
        "CancelledTransactions": ret_ratio * freq * 0.5,
        "ReturnRatio": ret_ratio, "TotalTransactions": freq,
        "UniqueInvoices": freq, "AvgLinesPerInvoice": 2.0,
        "Age": age, "SupportTicketsCount": support_tick,
        "SatisfactionScore": satisfaction,
        "AgeCategory": (0.0 if age < 30 else (1.0 if age < 50 else 2.0)),
        "SpendingCategory": (0.0 if (mon_avg*freq) < 300 else (1.0 if (mon_avg*freq) < 1000 else 2.0)),
        "FavoriteSeason": 1.5, "PreferredTimeOfDay": 1.0,
        "Region": 0.0, "WeekendPreference": 0.0,
        "BasketSizeCategory": (0.0 if mon_avg < 50 else (1.0 if mon_avg < 200 else 2.0)),
        "ProductDiversity": max(freq * 1.5, 1.0) / max(freq * 2.0, 1),
        "Gender": 0.0, "Country": 0.0,
        "RegYear": reg_year, "RegMonth": reg_month, "RegDay": 15.0,
        "RegWeekday": 2.0, "IP_IsPrivate": 1.0,
        "AvgBasket": (mon_avg * freq) / max(freq, 1),
        "ValuePerProduct": (mon_avg * freq) / max(max(freq * 1.5, 1.0) + 1, 1),
        "DiversityRatio": max(freq * 1.5, 1.0) / max(freq + 1, 1),
        "CancelRate": (ret_ratio * freq * 0.5) / max(freq + 1, 1),
    }

    cols = REGRESSOR_COLS if REGRESSOR_COLS else list(raw_map.keys())
    row  = {col: raw_map.get(col, 0.0) for col in cols}
    return pd.DataFrame([row])[cols]


def _align(df, model):
    if not hasattr(model, "feature_names_in_"):
        return df
    expected = list(model.feature_names_in_)
    for col in expected:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected]


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    feature_labels = FEATURE_NAMES if FEATURE_NAMES else []
    return render_template("index.html", feature_names=feature_labels)


@app.route("/debug-cols")
def debug_cols():
    return jsonify({
        "feature_names":    FEATURE_NAMES,
        "regressor_cols":   REGRESSOR_COLS,
        "models_loaded":    list(MODELS.keys()),
        "n_features":       len(FEATURE_NAMES),
        "n_regressor_cols": len(REGRESSOR_COLS),
        "train_medians_sample": dict(list(TRAIN_MEDIANS.items())[:5]),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        def _get(f, d):
            v = request.form.get(f, "").strip()
            try:    return float(v) if v else float(d)
            except: return float(d)

        # Collecter les inputs du formulaire
        # Les noms correspondent aux noms de features originaux
        u = {
            "Frequency":          _get("frequency",          5),
            "MonetaryTotal":      _get("monetary_total",    500),
            "MonetaryAvg":        _get("monetary_avg",      100),
            "CustomerTenureDays": _get("customer_tenure",   180),
            "ReturnRatio":        _get("return_ratio",     0.05),
            "SupportTicketsCount":_get("support_tickets",     1),
            "SatisfactionScore":  _get("satisfaction",      3.5),
            "Age":                _get("age",                35),
            "RegYear":            _get("reg_year",         2011),
            "RegMonth":           _get("reg_month",           6),
        }

        print("\n" + "="*60)
        print("[/predict]", u)
        result = {"error": None, "features": u}

        # ── 1. Pipeline Churn : features brutes → scaler → classifier ───────
        if "classifier_churn" in MODELS and "scaler" in MODELS:
            raw_df    = _build_feature_vector(u)
            raw_align = _align(raw_df.copy(), MODELS["scaler"])
            X_scaled  = MODELS["scaler"].transform(raw_align)
            X_sc_df   = pd.DataFrame(X_scaled, columns=raw_align.columns)

            clf   = MODELS["classifier_churn"]
            samp  = _align(X_sc_df.copy(), clf)
            churn_proba = float(clf.predict_proba(samp)[0, 1])
            churn_pred  = int(churn_proba >= 0.5)
            print(f"[CHURN] proba={churn_proba:.4f}  pred={churn_pred}")

            if   churn_proba < 0.25: risk, risk_color = "Faible",   "#27ae60"
            elif churn_proba < 0.50: risk, risk_color = "Moyen",    "#f39c12"
            elif churn_proba < 0.75: risk, risk_color = "Eleve",    "#e67e22"
            else:                    risk, risk_color = "Critique", "#e74c3c"

            result.update({
                "churn_proba": round(churn_proba * 100, 1),
                "churn_pred":  churn_pred,
                "risk":        risk,
                "risk_color":  risk_color,
            })

        # ── 2. Pipeline KMeans : features brutes → scaler → pca → kmeans ────
        if "pca" in MODELS and "kmeans" in MODELS and "scaler" in MODELS:
            raw_df    = _build_feature_vector(u)
            raw_align = _align(raw_df.copy(), MODELS["scaler"])
            X_scaled  = MODELS["scaler"].transform(raw_align)
            X_sc_df   = pd.DataFrame(X_scaled, columns=raw_align.columns)
            samp_pca  = _align(X_sc_df.copy(), MODELS["pca"])
            X_pca     = MODELS["pca"].transform(samp_pca)
            segment   = int(MODELS["kmeans"].predict(X_pca)[0])
            distances = MODELS["kmeans"].transform(X_pca)[0].tolist()
            print(f"[KMEANS] segment={segment}")
            result.update({
                "segment":       segment,
                "n_clusters":    int(MODELS["kmeans"].n_clusters),
                "seg_distances": {f"Cluster {i}": round(d, 3)
                                  for i, d in enumerate(distances)},
            })

        # ── 3. Régresseur MonetaryTotal ───────────────────────────────────
        if "regressor_monetary" in MODELS:
            raw_reg = _build_feature_vector_reg(u)
            samp    = _align(raw_reg.copy(), MODELS["regressor_monetary"])
            pred_m  = round(float(MODELS["regressor_monetary"].predict(samp)[0]), 2)
            print(f"[MONETARY] pred={pred_m}")
            result["monetary_pred"] = pred_m

        print("="*60 + "\n")
        return render_template("result.html", result=result)

    except Exception as e:
        print("[ERREUR /predict]\n" + traceback.format_exc())
        return render_template("result.html",
                               result={"error": str(e), "features": {}})


@app.route("/dashboard")
def dashboard():
    stats = {}
    churn_path = os.path.join(REPORTS_DIR, "churn_predictions.csv")
    if os.path.exists(churn_path):
        df = pd.read_csv(churn_path)
        stats["total_clients"] = len(df)
        if "Churn_Predicted" in df.columns:
            stats["churned"]    = int(df["Churn_Predicted"].sum())
            stats["churn_rate"] = round(df["Churn_Predicted"].mean() * 100, 1)
        if "Risk_Level" in df.columns:
            stats["risk_distribution"] = df["Risk_Level"].value_counts().to_dict()
    segment_path = os.path.join(REPORTS_DIR, "segment_predictions.csv")
    if os.path.exists(segment_path):
        df_seg     = pd.read_csv(segment_path)
        seg_counts = df_seg["Segment"].value_counts().sort_index().to_dict()
        stats["segment_distribution"] = {f"Cluster {k}": v for k, v in seg_counts.items()}
    return render_template("dashboard.html", stats=stats)


@app.route("/test-models")
def test_models():
    model_names  = ["classifier_churn", "scaler", "pca", "kmeans", "regressor_monetary"]
    model_status = {name: (name in MODELS) for name in model_names}
    return render_template("test_models.html",
                           model_status=model_status,
                           models_dir=MODELS_DIR,
                           feature_names=FEATURE_NAMES)


@app.route("/test-models/run", methods=["POST"])
def test_models_run():
    data = request.get_json()
    if not data: return jsonify({"error": "Corps JSON manquant"}), 400
    model_name = data.get("model")
    if not model_name: return jsonify({"error": "Champ 'model' manquant"}), 400
    if model_name not in MODELS:
        return jsonify({"model": model_name, "status": "non_charge",
                        "error": f"'{model_name}' non chargé."}), 404

    f = data.get("features", {})
    u = {
        "Frequency":           float(f.get("Frequency",           5)),
        "MonetaryTotal":       float(f.get("MonetaryTotal",      500)),
        "MonetaryAvg":         float(f.get("MonetaryAvg",        100)),
        "CustomerTenureDays":  float(f.get("CustomerTenureDays", 180)),
        "ReturnRatio":         float(f.get("ReturnRatio",        0.05)),
        "SupportTicketsCount": float(f.get("SupportTicketsCount",   1)),
        "SatisfactionScore":   float(f.get("SatisfactionScore",   3.5)),
        "Age":                 float(f.get("Age",                  35)),
        "RegYear":             float(f.get("RegYear",            2011)),
        "RegMonth":            float(f.get("RegMonth",              6)),
    }

    try:
        result = {"model": model_name, "status": "ok"}

        if model_name == "classifier_churn":
            raw_df   = _build_feature_vector(u)
            raw_a    = _align(raw_df.copy(), MODELS["scaler"])
            X_scaled = MODELS["scaler"].transform(raw_a)
            X_sc_df  = pd.DataFrame(X_scaled, columns=raw_a.columns)
            clf      = MODELS["classifier_churn"]
            samp     = _align(X_sc_df.copy(), clf)
            proba    = float(clf.predict_proba(samp)[0, 1])
            risk     = ("Faible" if proba < 0.25 else "Moyen" if proba < 0.50
                        else "Eleve" if proba < 0.75 else "Critique")
            result.update({"churn_probability": round(proba, 4),
                           "churn_predicted": int(proba >= 0.5),
                           "risk_level": risk,
                           "features_used": int(samp.shape[1]),
                           "feature_names": list(samp.columns)})

        elif model_name == "scaler":
            raw_df = _build_feature_vector(u)
            samp   = _align(raw_df.copy(), MODELS["scaler"])
            t      = MODELS["scaler"].transform(samp)
            result.update({"features_used": int(samp.shape[1]),
                           "feature_names": list(samp.columns),
                           "mean_before": round(float(samp.values.mean()), 4),
                           "mean_after":  round(float(t.mean()), 4),
                           "sample_values_transformed": [round(v, 4) for v in t[0, :8].tolist()]})

        elif model_name == "pca":
            raw_df   = _build_feature_vector(u)
            raw_a    = _align(raw_df.copy(), MODELS["scaler"])
            X_scaled = MODELS["scaler"].transform(raw_a)
            X_sc_df  = pd.DataFrame(X_scaled, columns=raw_a.columns)
            samp     = _align(X_sc_df.copy(), MODELS["pca"])
            X_pca    = MODELS["pca"].transform(samp)
            expl     = MODELS["pca"].explained_variance_ratio_.tolist()
            result.update({"n_components": int(MODELS["pca"].n_components_),
                           "components": [round(v, 4) for v in X_pca[0].tolist()],
                           "total_variance_explained": round(float(sum(expl)) * 100, 2)})

        elif model_name == "kmeans":
            raw_df   = _build_feature_vector(u)
            raw_a    = _align(raw_df.copy(), MODELS["scaler"])
            X_scaled = MODELS["scaler"].transform(raw_a)
            X_sc_df  = pd.DataFrame(X_scaled, columns=raw_a.columns)
            samp_pca = _align(X_sc_df.copy(), MODELS["pca"])
            X_pca    = MODELS["pca"].transform(samp_pca)
            km       = MODELS["kmeans"]
            seg      = int(km.predict(X_pca)[0])
            dists    = km.transform(X_pca)[0].tolist()
            result.update({"predicted_segment": seg,
                           "n_clusters": int(km.n_clusters),
                           "distances_to_centers": {f"Cluster {i}": round(d, 4)
                                                    for i, d in enumerate(dists)}})

        elif model_name == "regressor_monetary":
            raw_reg = _build_feature_vector_reg(u)
            samp    = _align(raw_reg.copy(), MODELS["regressor_monetary"])
            pred    = round(float(MODELS["regressor_monetary"].predict(samp)[0]), 2)
            result.update({"predicted_monetary": pred, "unit": "GBP",
                           "features_used": int(samp.shape[1])})

        return jsonify(result)

    except Exception as e:
        print("[ERREUR /test-models/run]\n" + traceback.format_exc())
        return jsonify({"model": model_name, "status": "erreur", "error": str(e)}), 500


@app.route("/test-models/status")
def test_models_status():
    model_names = ["classifier_churn", "scaler", "pca", "kmeans", "regressor_monetary"]
    return jsonify({"models": {n: n in MODELS for n in model_names},
                    "models_dir": MODELS_DIR,
                    "feature_names": FEATURE_NAMES,
                    "total_loaded": sum(1 for n in model_names if n in MODELS)})


if __name__ == "__main__":
    print("[RUN] Démarrage Flask — Retail ML")
    print("   URL       : http://127.0.0.1:5000")
    print("   Diagnostic: http://127.0.0.1:5000/debug-cols")
    app.run(debug=True, host="0.0.0.0", port=5000)