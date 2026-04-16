# =============================================================================
# flask_api.py — API Flask pour n8n
# Tous les endpoints dont n8n a besoin
# URL n8n : http://host.docker.internal:5000/...
# =============================================================================
from flask import Flask, jsonify, request
import subprocess, os, pandas as pd, json

app = Flask(__name__)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
MODELS_DIR  = os.path.join(SCRIPT_DIR, 'models')

# Chemins des scripts (certains sont dans le dossier parent)
SCRIPTS_PATHS = {
    '02_regression.py':      os.path.join(SCRIPT_DIR, '02_regression.py'),
    '03_classification.py':  os.path.join(SCRIPT_DIR, '03_classification.py'),
    '04_clustering.py':      os.path.join(SCRIPT_DIR, '04_clustering.py'),
    '05_timeseries.py':      os.path.join(os.path.dirname(SCRIPT_DIR), '05_timeseries.py'),
    '06_anomaly_detection.py': os.path.join(SCRIPT_DIR, '06_anomaly_detection.py'),
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER : lancer un script Python
# ─────────────────────────────────────────────────────────────────────────────
def run_script(script_name):
    script_path = SCRIPTS_PATHS.get(script_name,
                  os.path.join(SCRIPT_DIR, script_name))
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    try:
        result = subprocess.run(
            [VENV_PYTHON, script_path],
            capture_output=True, text=True,
            timeout=300, cwd=SCRIPT_DIR,
            encoding='utf-8', errors='replace',
            env=env
        )
        if result.returncode == 0:
            return {"status": "success", "script": script_name,
                    "output": result.stdout[-500:]}
        else:
            return {"status": "error", "script": script_name,
                    "error": result.stderr[-500:]}
    except Exception as e:
        return {"status": "error", "script": script_name, "error": str(e)}

# HELPER : lire un CSV et retourner en JSON
def read_csv(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.to_dict(orient='records')
    return []

# =============================================================================
# ENDPOINTS — EXÉCUTION DES SCRIPTS
# =============================================================================

@app.route('/predict/regression', methods=['POST'])
def regression():
    return jsonify(run_script('02_regression.py'))

@app.route('/predict/classification', methods=['POST'])
def classification():
    return jsonify(run_script('03_classification.py'))

@app.route('/predict/clustering', methods=['POST'])
def clustering():
    return jsonify(run_script('04_clustering.py'))

@app.route('/predict/timeseries', methods=['POST'])
def timeseries():
    return jsonify(run_script('05_timeseries.py'))

@app.route('/predict/anomaly', methods=['POST'])
def anomaly():
    return jsonify(run_script('06_anomaly_detection.py'))

# Lance TOUS les scripts en séquence
@app.route('/run/all', methods=['POST'])
def run_all():
    scripts = ['02_regression.py', '03_classification.py',
               '04_clustering.py', '05_timeseries.py',
               '06_anomaly_detection.py']
    results = []
    for s in scripts:
        r = run_script(s)
        results.append(r)
        if r['status'] == 'error':
            return jsonify({"status": "error", "failed_at": s,
                            "results": results}), 500
    return jsonify({"status": "success", "scripts_run": len(scripts),
                    "results": results})

# =============================================================================
# ENDPOINTS — LECTURE DES RÉSULTATS (pour n8n)
# =============================================================================

@app.route('/results/regression', methods=['GET'])
def results_regression():
    return jsonify({
        "forecasts":   read_csv('predictions_2026_2027.csv'),
        "comparison":  read_csv('regression_comparison.csv')
    })

@app.route('/results/classification', methods=['GET'])
def results_classification():
    return jsonify({
        "dropout_predictions": read_csv('predictions_dropout_2026.csv'),
        "comparison":          read_csv('classification_comparison.csv')
    })

@app.route('/results/clustering', methods=['GET'])
def results_clustering():
    return jsonify({
        "cluster_profile": read_csv('cluster_profile.csv'),
        "comparison":      read_csv('clustering_comparison.csv')
    })

@app.route('/results/timeseries', methods=['GET'])
def results_timeseries():
    return jsonify({
        "forecasts":  read_csv('forecasts_ts_2026_2027.csv'),
        "comparison": read_csv('timeseries_comparison.csv')
    })

@app.route('/results/anomaly', methods=['GET'])
def results_anomaly():
    data = read_csv('anomaly_results.csv')
    anomalies = [r for r in data if r.get('consensus', 0) == 1]
    return jsonify({
        "all_results":        data,
        "confirmed_anomalies": anomalies,
        "total_anomalies":    len(anomalies)
    })

# =============================================================================
# ENDPOINT PRINCIPAL — DASHBOARD COMPLET (1 seul appel n8n)
# =============================================================================

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Retourne TOUT en un seul appel — idéal pour n8n"""
    forecasts   = read_csv('predictions_2026_2027.csv')
    dropout     = read_csv('predictions_dropout_2026.csv')
    anomalies   = read_csv('anomaly_results.csv')
    ts_forecast = read_csv('forecasts_ts_2026_2027.csv')
    reg_cmp     = read_csv('regression_comparison.csv')
    cls_cmp     = read_csv('classification_comparison.csv')

    # Calcul alertes
    dropout_units  = [r for r in dropout  if r.get('dropout_risk_pred', 0) == 1]
    anomaly_units  = [r for r in anomalies if r.get('consensus', 0) == 1]
    total_forecast = sum(r.get('XGBoost_2026/2027', 0) for r in ts_forecast)

    has_alert = len(dropout_units) > 0 or len(anomaly_units) > 0

    alert_message = ""
    if dropout_units:
        units = [str(int(r['fk_type_unite'])) for r in dropout_units]
        alert_message += f"⚠️ Unités à risque dropout : {', '.join(units)}. "
    if anomaly_units:
        units = [f"U{int(r['fk_type_unite'])} ({r['season']})" for r in anomaly_units]
        alert_message += f"🚨 Anomalies confirmées : {', '.join(units)}."

    return jsonify({
        "status":             "success",
        "has_alert":          has_alert,
        "alert_message":      alert_message,
        "total_forecast_2026": round(total_forecast, 1),
        "dropout_units":      dropout_units,
        "anomaly_units":      anomaly_units,
        "forecasts":          forecasts,
        "ts_forecasts":       ts_forecast,
        "best_models": {
            "regression":     "Ridge (R²=0.770)",
            "classification": "Logistic Regression (F1=0.800)",
            "clustering":     "K-Means (Silhouette=0.6276)",
            "timeseries":     "XGBoost TS (R²=0.685)"
        },
        "metrics": {
            "regression":     reg_cmp,
            "classification": cls_cmp
        }
    })

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "ML Scout API running"})

# =============================================================================
if __name__ == '__main__':
    print("🚀 ML Scout API démarrée sur http://0.0.0.0:5000")
    print("   Endpoints disponibles :")
    print("   POST /run/all                → Lance tous les scripts")
    print("   GET  /dashboard              → Résultats complets")
    print("   GET  /results/regression     → Forecasts régression")
    print("   GET  /results/classification → Prédictions dropout")
    print("   GET  /results/anomaly        → Anomalies détectées")
    print("   GET  /results/timeseries     → Forecasts time series")
    print("   GET  /health                 → Vérification serveur")
    app.run(host='0.0.0.0', port=5000, debug=False)
