# =============================================================================
# flask_api.py — API Flask ML Scout avec Swagger UI
# Swagger UI disponible sur : http://localhost:5000/apidocs
# =============================================================================
from flask import Flask, jsonify, request
from flasgger import Swagger
import subprocess, os, pandas as pd, json

app = Flask(__name__)

# =============================================================================
# CONFIGURATION SWAGGER
# =============================================================================
swagger_config = {
    "headers": [],
    "specs": [{"endpoint": "apispec", "route": "/apispec.json"}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs"
}

swagger_template = {
    "info": {
        "title": "ML Scout API",
        "description": "API de prediction ML pour le groupe scout — regression, classification, clustering, anomalies, time series",
        "version": "1.0.0",
        "contact": {"name": "ML Scout Team"}
    },
    "tags": [
        {"name": "Health",      "description": "Verification du serveur"},
        {"name": "Predictions", "description": "Endpoints de prediction ML en temps reel"},
        {"name": "Resultats",   "description": "Lecture des resultats sauvegardes"},
        {"name": "Execution",   "description": "Lancement des scripts ML"},
        {"name": "Dashboard",   "description": "Vue complete des donnees et rapport"}
    ]
}

Swagger(app, config=swagger_config, template=swagger_template)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
MODELS_DIR  = os.path.join(SCRIPT_DIR, 'models')

SCRIPTS_PATHS = {
    '02_regression.py':        os.path.join(SCRIPT_DIR, '02_regression.py'),
    '03_classification.py':    os.path.join(SCRIPT_DIR, '03_classification.py'),
    '04_clustering.py':        os.path.join(SCRIPT_DIR, '04_clustering.py'),
    '05_timeseries.py':        os.path.join(os.path.dirname(SCRIPT_DIR), '05_timeseries.py'),
    '06_anomaly_detection.py': os.path.join(SCRIPT_DIR, '06_anomaly_detection.py'),
}

# =============================================================================
# HELPERS
# =============================================================================
def run_script(script_name):
    script_path = SCRIPTS_PATHS.get(script_name, os.path.join(SCRIPT_DIR, script_name))
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    try:
        result = subprocess.run(
            [VENV_PYTHON, script_path],
            capture_output=True, text=True,
            timeout=300, cwd=SCRIPT_DIR,
            encoding='utf-8', errors='replace', env=env
        )
        if result.returncode == 0:
            return {"status": "success", "script": script_name, "output": result.stdout[-500:]}
        else:
            return {"status": "error", "script": script_name, "error": result.stderr[-500:]}
    except Exception as e:
        return {"status": "error", "script": script_name, "error": str(e)}

def read_csv(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path).to_dict(orient='records')
    return []

# =============================================================================
# HEALTH
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    """
    Verification du serveur
    ---
    tags:
      - Health
    responses:
      200:
        description: Serveur operationnel
        schema:
          properties:
            status:
              type: string
              example: ok
            message:
              type: string
              example: ML Scout API running
            version:
              type: string
              example: 1.0.0
            endpoints:
              type: array
              items:
                type: string
    """
    return jsonify({
        "status": "ok",
        "message": "ML Scout API running",
        "version": "1.0.0",
        "swagger_ui": "http://localhost:5000/apidocs",
        "endpoints": [
            "POST /predict                → Prediction unifiee (regression/classification)",
            "POST /run/all               → Lance tous les scripts ML",
            "POST /run/mlops             → Lance le pipeline MLOps",
            "GET  /dashboard             → Resultats complets",
            "GET  /report/html           → Rapport HTML enrichi",
            "GET  /results/regression    → Forecasts regression",
            "GET  /results/classification→ Predictions dropout",
            "GET  /results/clustering    → Profils clustering",
            "GET  /results/timeseries    → Forecasts time series",
            "GET  /results/anomaly       → Anomalies detectees",
            "GET  /health                → Verification serveur"
        ]
    })

# =============================================================================
# PREDICTION UNIFIEE
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction unifiee — regression ou classification
    ---
    tags:
      - Predictions
    parameters:
      - in: body
        name: body
        required: true
        schema:
          required:
            - type
            - fk_type_unite
          properties:
            type:
              type: string
              enum: [regression, classification]
              example: regression
              description: "regression = nombre de membres prevu | classification = risque dropout"
            fk_type_unite:
              type: integer
              example: 1
              description: "Numero de l'unite scout (1 a 6)"
            members_lag1:
              type: number
              example: 12
              description: "Nombre de membres actuels (optionnel — utilise les donnees historiques si absent)"
            leak_rate_lag1:
              type: number
              example: 52.0
              description: "Taux de fuite en % (optionnel)"
    responses:
      200:
        description: Prediction reussie
        schema:
          properties:
            type:
              type: string
              example: regression
            model:
              type: string
              example: Ridge Regression (R2=0.770)
            prediction:
              type: number
              example: 14.5
            unit:
              type: string
              example: Unite 1
            season:
              type: string
              example: 2026/2027
            message:
              type: string
              example: "Prediction : 14.5 adherents pour 2026/2027"
      400:
        description: Donnees manquantes ou type invalide
      500:
        description: Erreur interne
    """
    import joblib, numpy as np

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    pred_type = data.get('type', 'regression')
    unit      = int(data.get('fk_type_unite', 1))

    df_ts  = pd.read_csv(os.path.join(SCRIPT_DIR, 'data', 'dataset_timeseries.csv'))
    df_clu = pd.read_csv(os.path.join(SCRIPT_DIR, 'data', 'dataset_clustering.csv'))
    df_clu['season_year'] = df_clu['season'].str[:4].astype(int)
    df = df_ts.merge(df_clu[['fk_type_unite','season_year','leak_rate']],
                     on=['fk_type_unite','season_year'], how='left')
    df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

    sub  = df[df['fk_type_unite'] == unit].sort_values('season_year')
    last = sub.iloc[-1] if len(sub) > 0 else None
    prev = sub.iloc[-2] if len(sub) >= 2 else sub.iloc[-1]

    m_curr   = float(data.get('members_lag1', last['nbmr_members_season'] if last is not None else 10))
    leak1    = float(data.get('leak_rate_lag1', last['leak_rate'] if last is not None else 0))
    m_prev   = float(prev['nbmr_members_season']) if prev is not None else m_curr
    growth   = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta    = m_curr - m_prev
    ret      = 1 - leak1 / 100
    avg_past = float(sub['nbmr_members_season'].mean()) if len(sub) > 0 else m_curr

    FEAT = ['members_lag1','leak_rate_lag1','growth_rate_lag1','delta_members_lag1',
            'avg_members_unite_past','season_index','fk_type_unite','retention_rate','members_x_retention']
    row = {
        'members_lag1': m_curr, 'leak_rate_lag1': leak1, 'growth_rate_lag1': growth,
        'delta_members_lag1': delta, 'avg_members_unite_past': avg_past,
        'season_index': 4, 'fk_type_unite': unit, 'retention_rate': ret,
        'members_x_retention': m_curr * ret,
    }
    X = pd.DataFrame([row])[FEAT].values

    try:
        if pred_type == 'regression':
            model = joblib.load(os.path.join(MODELS_DIR, 'regressor_ridge.pkl'))
            pred  = max(float(model.predict(X)[0]), 0)
            return jsonify({
                "type": "regression", "model": "Ridge Regression (R2=0.770)",
                "input": {"fk_type_unite": unit, "members_lag1": m_curr, "leak_rate_lag1": leak1},
                "prediction": round(pred, 1), "unit": f"Unite {unit}", "season": "2026/2027",
                "message": f"Prediction : {round(pred,1)} adherents pour 2026/2027"
            })
        elif pred_type in ['classification', 'dropout']:
            model = joblib.load(os.path.join(MODELS_DIR, 'classifier_lr.pkl'))
            pred  = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0][1])
            return jsonify({
                "type": "classification", "model": "Logistic Regression (F1=0.800)",
                "input": {"fk_type_unite": unit, "members_lag1": m_curr, "leak_rate_lag1": leak1},
                "prediction": pred, "probability": round(proba, 3),
                "dropout_risk": "OUI" if pred == 1 else "NON", "unit": f"Unite {unit}",
                "message": f"Risque dropout : {'OUI' if pred==1 else 'NON'} (P={round(proba*100,1)}%)"
            })
        else:
            return jsonify({"error": "Type invalide. Utilisez 'regression' ou 'classification'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================================================================
# EXECUTION DES SCRIPTS
# =============================================================================
@app.route('/run/all', methods=['POST'])
def run_all():
    """
    Lance tous les scripts ML en sequence
    ---
    tags:
      - Execution
    responses:
      200:
        description: Tous les scripts executes avec succes
        schema:
          properties:
            status:
              type: string
              example: success
            scripts_run:
              type: integer
              example: 5
      500:
        description: Erreur sur un des scripts
    """
    scripts = ['02_regression.py','03_classification.py','04_clustering.py',
               '05_timeseries.py','06_anomaly_detection.py']
    results = []
    for s in scripts:
        r = run_script(s)
        results.append(r)
        if r['status'] == 'error':
            return jsonify({"status": "error", "failed_at": s, "results": results}), 500
    return jsonify({"status": "success", "scripts_run": len(scripts), "results": results})

@app.route('/run/mlops', methods=['POST'])
def run_mlops():
    """
    Lance le pipeline MLOps (MLflow tracking + registry + staging)
    ---
    tags:
      - Execution
    responses:
      200:
        description: Pipeline MLOps execute avec succes
      500:
        description: Erreur pipeline
    """
    results = []
    for s in ['07_mlops_mlflow.py','08_mlops_registry.py','09_mlops_staging_production.py']:
        r = run_script(s)
        results.append(r)
        if r['status'] == 'error':
            return jsonify({"status": "error", "failed_at": s, "results": results}), 500
    return jsonify({"status": "success", "scripts_run": 3, "results": results})

@app.route('/predict/regression', methods=['POST'])
def run_regression():
    """
    Execute le script de regression complet
    ---
    tags:
      - Execution
    responses:
      200:
        description: Script regression execute
    """
    return jsonify(run_script('02_regression.py'))

@app.route('/predict/classification', methods=['POST'])
def run_classification():
    """
    Execute le script de classification complet
    ---
    tags:
      - Execution
    responses:
      200:
        description: Script classification execute
    """
    return jsonify(run_script('03_classification.py'))

@app.route('/predict/clustering', methods=['POST'])
def run_clustering():
    """
    Execute le script de clustering complet
    ---
    tags:
      - Execution
    responses:
      200:
        description: Script clustering execute
    """
    return jsonify(run_script('04_clustering.py'))

@app.route('/predict/timeseries', methods=['POST'])
def run_timeseries():
    """
    Execute le script de time series complet
    ---
    tags:
      - Execution
    responses:
      200:
        description: Script time series execute
    """
    return jsonify(run_script('05_timeseries.py'))

@app.route('/predict/anomaly', methods=['POST'])
def run_anomaly():
    """
    Execute le script de detection d'anomalies complet
    ---
    tags:
      - Execution
    responses:
      200:
        description: Script anomalies execute
    """
    return jsonify(run_script('06_anomaly_detection.py'))

# =============================================================================
# RESULTATS
# =============================================================================
@app.route('/results/regression', methods=['GET'])
def results_regression():
    """
    Resultats de regression — forecasts et comparaison des modeles
    ---
    tags:
      - Resultats
    responses:
      200:
        description: Forecasts 2026/2027 et metriques des modeles
        schema:
          properties:
            forecasts:
              type: array
              description: Predictions par unite pour 2026/2027
            comparison:
              type: array
              description: Comparaison MSE/RMSE/MAE/R2 des modeles
    """
    return jsonify({
        "forecasts":  read_csv('predictions_2026_2027.csv'),
        "comparison": read_csv('regression_comparison.csv')
    })

@app.route('/results/classification', methods=['GET'])
def results_classification():
    """
    Resultats de classification — predictions dropout et comparaison
    ---
    tags:
      - Resultats
    responses:
      200:
        description: Predictions de risque dropout par unite
        schema:
          properties:
            dropout_predictions:
              type: array
              description: Risque dropout par unite pour 2026
            comparison:
              type: array
              description: Comparaison Accuracy/F1/ROC-AUC des modeles
    """
    return jsonify({
        "dropout_predictions": read_csv('predictions_dropout_2026.csv'),
        "comparison":          read_csv('classification_comparison.csv')
    })

@app.route('/results/clustering', methods=['GET'])
def results_clustering():
    """
    Resultats de clustering — profils des clusters
    ---
    tags:
      - Resultats
    responses:
      200:
        description: Profils des clusters et comparaison des modeles
    """
    return jsonify({
        "cluster_profile": read_csv('cluster_profile.csv'),
        "comparison":      read_csv('clustering_comparison.csv')
    })

@app.route('/results/timeseries', methods=['GET'])
def results_timeseries():
    """
    Resultats de time series — forecasts et comparaison
    ---
    tags:
      - Resultats
    responses:
      200:
        description: Forecasts time series 2026/2027
    """
    return jsonify({
        "forecasts":  read_csv('forecasts_ts_2026_2027.csv'),
        "comparison": read_csv('timeseries_comparison.csv')
    })

@app.route('/results/anomaly', methods=['GET'])
def results_anomaly():
    """
    Resultats de detection d'anomalies
    ---
    tags:
      - Resultats
    responses:
      200:
        description: Anomalies detectees par consensus (vote >= 2/3 algorithmes)
        schema:
          properties:
            all_results:
              type: array
              description: Tous les resultats
            confirmed_anomalies:
              type: array
              description: Anomalies confirmees (consensus = 1)
            total_anomalies:
              type: integer
              example: 2
    """
    data      = read_csv('anomaly_results.csv')
    anomalies = [r for r in data if r.get('consensus', 0) == 1]
    return jsonify({
        "all_results":         data,
        "confirmed_anomalies": anomalies,
        "total_anomalies":     len(anomalies)
    })

# =============================================================================
# DASHBOARD
# =============================================================================
@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Dashboard complet — tous les resultats en un seul appel
    ---
    tags:
      - Dashboard
    responses:
      200:
        description: Vue complete (forecasts, anomalies, dropout, metriques, alertes)
        schema:
          properties:
            status:
              type: string
              example: success
            has_alert:
              type: boolean
              example: true
            alert_message:
              type: string
              example: "Unites a risque dropout : 2, 4"
            total_forecast_2026:
              type: number
              example: 87.0
            best_models:
              type: object
    """
    forecasts   = read_csv('predictions_2026_2027.csv')
    dropout     = read_csv('predictions_dropout_2026.csv')
    anomalies   = read_csv('anomaly_results.csv')
    ts_forecast = read_csv('forecasts_ts_2026_2027.csv')
    reg_cmp     = read_csv('regression_comparison.csv')
    cls_cmp     = read_csv('classification_comparison.csv')

    dropout_units  = [r for r in dropout   if r.get('dropout_risk_pred', 0) == 1]
    anomaly_units  = [r for r in anomalies if r.get('consensus', 0) == 1]
    total_forecast = sum(r.get('XGBoost_2026/2027', 0) for r in ts_forecast)
    has_alert      = len(dropout_units) > 0 or len(anomaly_units) > 0

    alert_message = ""
    if dropout_units:
        units = [str(int(r['fk_type_unite'])) for r in dropout_units]
        alert_message += f"Unites a risque dropout : {', '.join(units)}. "
    if anomaly_units:
        units = [f"U{int(r['fk_type_unite'])} ({r['season']})" for r in anomaly_units]
        alert_message += f"Anomalies confirmees : {', '.join(units)}."

    return jsonify({
        "status":              "success",
        "has_alert":           has_alert,
        "alert_message":       alert_message,
        "total_forecast_2026": round(total_forecast, 1),
        "dropout_units":       dropout_units,
        "anomaly_units":       anomaly_units,
        "forecasts":           forecasts,
        "ts_forecasts":        ts_forecast,
        "best_models": {
            "regression":     "Ridge (R2=0.770)",
            "classification": "Logistic Regression (F1=0.800)",
            "clustering":     "K-Means (Silhouette=0.6276)",
            "timeseries":     "XGBoost TS (R2=0.685)"
        },
        "metrics": {"regression": reg_cmp, "classification": cls_cmp}
    })

@app.route('/report/html', methods=['GET'])
def report_html():
    """
    Rapport HTML complet — pret a envoyer par email
    ---
    tags:
      - Dashboard
    responses:
      200:
        description: Rapport HTML genere
        schema:
          properties:
            html:
              type: string
              description: Contenu HTML du rapport
            has_alert:
              type: boolean
            total_forecast:
              type: number
            dropout_count:
              type: integer
            anomaly_count:
              type: integer
    """
    forecasts     = read_csv('predictions_2026_2027.csv')
    dropout       = read_csv('predictions_dropout_2026.csv')
    anomalies     = read_csv('anomaly_results.csv')
    ts_forecast   = read_csv('forecasts_ts_2026_2027.csv')
    reg_cmp       = read_csv('regression_comparison.csv')
    cls_cmp       = read_csv('classification_comparison.csv')

    dropout_units = [r for r in dropout   if r.get('dropout_risk_pred', 0) == 1]
    anomaly_units = [r for r in anomalies if r.get('consensus', 0) == 1]
    has_alert     = len(dropout_units) > 0 or len(anomaly_units) > 0
    total_fc      = sum(r.get('XGBoost_2026/2027', 0) for r in ts_forecast)

    alert_color = '#C62828' if has_alert else '#2E7D32'
    alert_icon  = 'ALERTE' if has_alert else 'OK'
    alert_text  = 'ALERTE — Action requise' if has_alert else 'Tout est normal'

    fc_rows = ''
    for r in ts_forecast:
        u     = int(r.get('Unite', 0))
        real  = r.get('Reel_2025/2026', '-')
        fc    = r.get('XGBoost_2026/2027', '-')
        d_r   = next((x for x in dropout if int(x.get('fk_type_unite', 0)) == u), {})
        proba = d_r.get('dropout_proba', 0)
        risk  = d_r.get('dropout_risk_pred', 0)
        badge = 'RISQUE' if risk == 1 else 'OK'
        fc_rows += f'<tr><td>{u}</td><td>{real}</td><td>{fc}</td><td>{round(float(proba)*100,1)}%</td><td>{badge}</td></tr>'

    reg_rows = ''
    for r in reg_cmp:
        reg_rows += f'<tr><td>{r.get("Modele","-")}</td><td>{r.get("MSE","-")}</td><td>{r.get("RMSE","-")}</td><td>{r.get("MAE","-")}</td><td>{r.get("R2","-")}</td></tr>'

    cls_rows = ''
    for r in cls_cmp:
        cls_rows += f'<tr><td>{r.get("Modele","-")}</td><td>{r.get("Accuracy","-")}</td><td>{r.get("F1-Score","-")}</td><td>{r.get("ROC-AUC","-")}</td></tr>'

    anom_rows = ''
    for r in anomaly_units:
        anom_rows += f'<tr><td>U{int(r.get("fk_type_unite",0))}</td><td>{r.get("season","-")}</td><td>{r.get("nbmr_members_season","-")}</td><td>{r.get("leak_rate","-")}%</td><td>{int(r.get("vote_sum",0))}/3</td></tr>'
    if not anom_rows:
        anom_rows = '<tr><td colspan="5">Aucune anomalie detectee</td></tr>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
th{{background:#1565C0;color:white;padding:10px}}
td{{padding:8px;border:1px solid #ddd}}
.alert{{background:{alert_color};color:white;padding:16px;border-radius:8px;margin-bottom:20px}}
h1{{color:#1565C0}}h2{{color:#1565C0}}</style></head>
<body>
<h1>Rapport ML Scout</h1>
<div class="alert"><b>{alert_icon} — {alert_text}</b></div>
<p>Forecast total 2026/2027 : <b>{round(total_fc,0):.0f}</b> membres | Anomalies : <b>{len(anomaly_units)}</b> | Dropout risk : <b>{len(dropout_units)}</b></p>
<h2>Forecasts par Unite</h2>
<table><tr><th>Unite</th><th>Reel 2025/26</th><th>Forecast 2026/27</th><th>P(Dropout)</th><th>Statut</th></tr>{fc_rows}</table>
<h2>Regression — Modeles</h2>
<table><tr><th>Modele</th><th>MSE</th><th>RMSE</th><th>MAE</th><th>R2</th></tr>{reg_rows}</table>
<h2>Classification — Modeles</h2>
<table><tr><th>Modele</th><th>Accuracy</th><th>F1-Score</th><th>ROC-AUC</th></tr>{cls_rows}</table>
<h2>Anomalies Detectees</h2>
<table><tr><th>Unite</th><th>Saison</th><th>Membres</th><th>Taux Fuite</th><th>Vote</th></tr>{anom_rows}</table>
</body></html>"""

    return jsonify({
        "html":           html,
        "has_alert":      has_alert,
        "alert_message":  f"{alert_icon} — {alert_text}",
        "total_forecast": round(total_fc, 1),
        "dropout_count":  len(dropout_units),
        "anomaly_count":  len(anomaly_units)
    })

# =============================================================================
if __name__ == '__main__':
    print("ML Scout API sur http://0.0.0.0:5000")
    print("Swagger UI : http://localhost:5000/apidocs")
    app.run(host='0.0.0.0', port=5000, debug=False)
