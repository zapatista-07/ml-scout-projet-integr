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
# ENDPOINT RAPPORT HTML ENRICHI (pour n8n Node Code)
# =============================================================================

@app.route('/report/html', methods=['GET'])
def report_html():
    """Génère un rapport HTML complet — prêt à envoyer par email"""
    forecasts   = read_csv('predictions_2026_2027.csv')
    dropout     = read_csv('predictions_dropout_2026.csv')
    anomalies   = read_csv('anomaly_results.csv')
    ts_forecast = read_csv('forecasts_ts_2026_2027.csv')
    reg_cmp     = read_csv('regression_comparison.csv')
    cls_cmp     = read_csv('classification_comparison.csv')
    clu_cmp     = read_csv('clustering_comparison.csv')
    ts_cmp      = read_csv('timeseries_comparison.csv')

    dropout_units = [r for r in dropout  if r.get('dropout_risk_pred', 0) == 1]
    anomaly_units = [r for r in anomalies if r.get('consensus', 0) == 1]
    has_alert     = len(dropout_units) > 0 or len(anomaly_units) > 0
    total_fc      = sum(r.get('XGBoost_2026/2027', 0) for r in ts_forecast)

    # Couleur alerte
    alert_color  = '#C62828' if has_alert else '#2E7D32'
    alert_icon   = '🚨' if has_alert else '✅'
    alert_text   = 'ALERTE — Action requise' if has_alert else 'Tout est normal'

    # Tableau forecasts
    fc_rows = ''
    for r in ts_forecast:
        u    = int(r.get('Unité', 0))
        real = r.get('Réel_2025/2026', '-')
        fc   = r.get('XGBoost_2026/2027', '-')
        d_r  = next((x for x in dropout if int(x.get('fk_type_unite',0))==u), {})
        proba = d_r.get('dropout_proba', 0)
        risk  = d_r.get('dropout_risk_pred', 0)
        risk_badge = f'<span style="color:#C62828;font-weight:bold">🔴 Risque</span>' \
                     if risk == 1 else '<span style="color:#2E7D32">🟢 OK</span>'
        fc_rows += f"""
        <tr>
          <td style="padding:8px;border:1px solid #ddd;text-align:center"><b>U{u}</b></td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{real}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center"><b>{fc}</b></td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{round(float(proba)*100,1)}%</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{risk_badge}</td>
        </tr>"""

    # Tableau métriques régression
    reg_rows = ''
    for r in reg_cmp:
        best = r.get('R²', 0) == max(x.get('R²', 0) for x in reg_cmp)
        bg   = 'background:#E8F5E9' if best else ''
        reg_rows += f"""
        <tr style="{bg}">
          <td style="padding:8px;border:1px solid #ddd">{r.get('Modèle','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('MSE','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('RMSE','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('MAE','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center"><b>{r.get('R²','-')}</b></td>
        </tr>"""

    # Tableau métriques classification
    cls_rows = ''
    for r in cls_cmp:
        best = r.get('F1-Score', 0) == max(x.get('F1-Score', 0) for x in cls_cmp)
        bg   = 'background:#E8F5E9' if best else ''
        cls_rows += f"""
        <tr style="{bg}">
          <td style="padding:8px;border:1px solid #ddd">{r.get('Modèle','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('Accuracy','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('Precision','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('Recall','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center"><b>{r.get('F1-Score','-')}</b></td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('ROC-AUC','-')}</td>
        </tr>"""

    # Anomalies
    anom_rows = ''
    for r in anomaly_units:
        anom_rows += f"""
        <tr style="background:#FFEBEE">
          <td style="padding:8px;border:1px solid #ddd">U{int(r.get('fk_type_unite',0))}</td>
          <td style="padding:8px;border:1px solid #ddd">{r.get('season','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('nbmr_members_season','-')}</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{r.get('leak_rate','-')}%</td>
          <td style="padding:8px;border:1px solid #ddd;text-align:center">{int(r.get('vote_sum',0))}/3</td>
        </tr>"""
    if not anom_rows:
        anom_rows = '<tr><td colspan="5" style="padding:8px;text-align:center;color:#2E7D32">✅ Aucune anomalie détectée</td></tr>'

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;color:#333">

  <!-- HEADER -->
  <div style="background:linear-gradient(135deg,#1565C0,#0D47A1);padding:24px;border-radius:12px;color:white;margin-bottom:24px">
    <h1 style="margin:0;font-size:1.6rem">🏆 Rapport Mensuel ML Scout</h1>
    <p style="margin:8px 0 0;opacity:0.85">Pipeline ML Automatisé — Groupe Scout</p>
  </div>

  <!-- ALERTE -->
  <div style="background:{alert_color};color:white;padding:16px;border-radius:8px;margin-bottom:24px;font-size:1.1rem">
    {alert_icon} <b>{alert_text}</b>
  </div>

  <!-- KPIs -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:16px;margin-bottom:24px">
    <div style="background:#E3F2FD;padding:16px;border-radius:8px;text-align:center;border-left:4px solid #1565C0">
      <div style="font-size:0.75rem;color:#666;text-transform:uppercase">Total 2025/26</div>
      <div style="font-size:1.8rem;font-weight:bold;color:#1565C0">83</div>
    </div>
    <div style="background:#E8F5E9;padding:16px;border-radius:8px;text-align:center;border-left:4px solid #2E7D32">
      <div style="font-size:0.75rem;color:#666;text-transform:uppercase">Forecast 2026/27</div>
      <div style="font-size:1.8rem;font-weight:bold;color:#2E7D32">{round(total_fc,0):.0f}</div>
    </div>
    <div style="background:#FFEBEE;padding:16px;border-radius:8px;text-align:center;border-left:4px solid #C62828">
      <div style="font-size:0.75rem;color:#666;text-transform:uppercase">Anomalies</div>
      <div style="font-size:1.8rem;font-weight:bold;color:#C62828">{len(anomaly_units)}</div>
    </div>
    <div style="background:#FFF8E1;padding:16px;border-radius:8px;text-align:center;border-left:4px solid #F57F17">
      <div style="font-size:0.75rem;color:#666;text-transform:uppercase">Dropout Risk</div>
      <div style="font-size:1.8rem;font-weight:bold;color:#F57F17">{len(dropout_units)}</div>
    </div>
  </div>

  <!-- FORECASTS PAR UNITÉ -->
  <h2 style="color:#1565C0;border-bottom:2px solid #E0E7EF;padding-bottom:8px">📊 Forecasts 2026/2027 par Unité</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#1565C0;color:white">
      <th style="padding:10px;border:1px solid #ddd">Unité</th>
      <th style="padding:10px;border:1px solid #ddd">Réel 2025/26</th>
      <th style="padding:10px;border:1px solid #ddd">Forecast 2026/27</th>
      <th style="padding:10px;border:1px solid #ddd">P(Dropout)</th>
      <th style="padding:10px;border:1px solid #ddd">Statut</th>
    </tr>
    {fc_rows}
  </table>

  <!-- MÉTRIQUES RÉGRESSION -->
  <h2 style="color:#1565C0;border-bottom:2px solid #E0E7EF;padding-bottom:8px">📈 Régression — Comparaison Modèles</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#1565C0;color:white">
      <th style="padding:10px;border:1px solid #ddd">Modèle</th>
      <th style="padding:10px;border:1px solid #ddd">MSE</th>
      <th style="padding:10px;border:1px solid #ddd">RMSE</th>
      <th style="padding:10px;border:1px solid #ddd">MAE</th>
      <th style="padding:10px;border:1px solid #ddd">R²</th>
    </tr>
    {reg_rows}
  </table>

  <!-- MÉTRIQUES CLASSIFICATION -->
  <h2 style="color:#1565C0;border-bottom:2px solid #E0E7EF;padding-bottom:8px">🎯 Classification — Comparaison Modèles</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#1565C0;color:white">
      <th style="padding:10px;border:1px solid #ddd">Modèle</th>
      <th style="padding:10px;border:1px solid #ddd">Accuracy</th>
      <th style="padding:10px;border:1px solid #ddd">Precision</th>
      <th style="padding:10px;border:1px solid #ddd">Recall</th>
      <th style="padding:10px;border:1px solid #ddd">F1-Score</th>
      <th style="padding:10px;border:1px solid #ddd">ROC-AUC</th>
    </tr>
    {cls_rows}
  </table>

  <!-- ANOMALIES -->
  <h2 style="color:#C62828;border-bottom:2px solid #FFCDD2;padding-bottom:8px">🚨 Anomalies Détectées (Consensus ≥2/3)</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#C62828;color:white">
      <th style="padding:10px;border:1px solid #ddd">Unité</th>
      <th style="padding:10px;border:1px solid #ddd">Saison</th>
      <th style="padding:10px;border:1px solid #ddd">Membres</th>
      <th style="padding:10px;border:1px solid #ddd">Taux Fuite</th>
      <th style="padding:10px;border:1px solid #ddd">Vote</th>
    </tr>
    {anom_rows}
  </table>

  <!-- MEILLEURS MODÈLES -->
  <h2 style="color:#1565C0;border-bottom:2px solid #E0E7EF;padding-bottom:8px">🏆 Meilleurs Modèles (Production)</h2>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px">
    <tr style="background:#1565C0;color:white">
      <th style="padding:10px;border:1px solid #ddd">Section</th>
      <th style="padding:10px;border:1px solid #ddd">Modèle</th>
      <th style="padding:10px;border:1px solid #ddd">Métrique</th>
    </tr>
    <tr style="background:#E8F5E9">
      <td style="padding:8px;border:1px solid #ddd">D — Régression</td>
      <td style="padding:8px;border:1px solid #ddd">Ridge Regression</td>
      <td style="padding:8px;border:1px solid #ddd"><b>R²=0.770</b></td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd">C — Classification</td>
      <td style="padding:8px;border:1px solid #ddd">Logistic Regression</td>
      <td style="padding:8px;border:1px solid #ddd"><b>F1=0.800</b></td>
    </tr>
    <tr style="background:#E8F5E9">
      <td style="padding:8px;border:1px solid #ddd">E — Clustering</td>
      <td style="padding:8px;border:1px solid #ddd">K-Means + GMM</td>
      <td style="padding:8px;border:1px solid #ddd"><b>Silhouette=0.6276</b></td>
    </tr>
    <tr>
      <td style="padding:8px;border:1px solid #ddd">F — Time Series</td>
      <td style="padding:8px;border:1px solid #ddd">XGBoost TS</td>
      <td style="padding:8px;border:1px solid #ddd"><b>R²=0.685</b></td>
    </tr>
  </table>

  <!-- FOOTER -->
  <div style="background:#F5F5F5;padding:16px;border-radius:8px;text-align:center;color:#888;font-size:0.8rem">
    Généré automatiquement par <b>ML Scout Pipeline</b> — n8n + Flask + MLflow<br>
    Données DWH — 18 observations — 6 unités × 3 saisons
  </div>

</body>
</html>"""

    return jsonify({
        "html": html,
        "has_alert": has_alert,
        "alert_message": f"{alert_icon} {alert_text}",
        "total_forecast": round(total_fc, 1),
        "dropout_count": len(dropout_units),
        "anomaly_count": len(anomaly_units)
    })


# =============================================================================
# ENDPOINT MLOps — Lance les scripts MLflow
# =============================================================================

@app.route('/run/mlops', methods=['POST'])
def run_mlops():
    results = []
    for s in ['07_mlops_mlflow.py', '08_mlops_registry.py', '09_mlops_staging_production.py']:
        r = run_script(s)
        results.append(r)
        if r['status'] == 'error':
            return jsonify({"status": "error", "failed_at": s, "results": results}), 500
    return jsonify({"status": "success", "scripts_run": 3, "results": results})

# =============================================================================
# ENDPOINT /predict — UNIFIÉ (grille MLOps)
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint unifié de prédiction.
    Input JSON :
      { "type": "regression" | "classification", "fk_type_unite": 1,
        "members_lag1": 12, "leak_rate_lag1": 52.0 }
    Output JSON : { "prediction": ..., "model": ..., "message": ... }
    """
    import joblib
    import numpy as np

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

    m_curr = float(data.get('members_lag1',
                   last['nbmr_members_season'] if last is not None else 10))
    leak1  = float(data.get('leak_rate_lag1',
                   last['leak_rate'] if last is not None else 0))
    m_prev = float(prev['nbmr_members_season']) if prev is not None else m_curr
    growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta  = m_curr - m_prev
    ret    = 1 - leak1 / 100
    avg_past = float(sub['nbmr_members_season'].mean()) if len(sub) > 0 else m_curr

    FEAT = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
            'delta_members_lag1', 'avg_members_unite_past',
            'season_index', 'fk_type_unite',
            'retention_rate', 'members_x_retention']

    row = {
        'members_lag1': m_curr, 'leak_rate_lag1': leak1,
        'growth_rate_lag1': growth, 'delta_members_lag1': delta,
        'avg_members_unite_past': avg_past, 'season_index': 4,
        'fk_type_unite': unit, 'retention_rate': ret,
        'members_x_retention': m_curr * ret,
    }
    X = pd.DataFrame([row])[FEAT].values

    try:
        if pred_type == 'regression':
            model = joblib.load(os.path.join(MODELS_DIR, 'regressor_ridge.pkl'))
            pred  = max(float(model.predict(X)[0]), 0)
            return jsonify({
                "type": "regression", "model": "Ridge Regression (R2=0.770)",
                "input": {"fk_type_unite": unit, "members_lag1": m_curr,
                          "leak_rate_lag1": leak1},
                "prediction": round(pred, 1), "unit": f"Unite {unit}",
                "season": "2026/2027",
                "message": f"Prediction : {round(pred,1)} adherents pour 2026/2027"
            })
        elif pred_type in ['classification', 'dropout']:
            model = joblib.load(os.path.join(MODELS_DIR, 'classifier_lr.pkl'))
            pred  = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0][1])
            return jsonify({
                "type": "classification", "model": "Logistic Regression (F1=0.800)",
                "input": {"fk_type_unite": unit, "members_lag1": m_curr,
                          "leak_rate_lag1": leak1},
                "prediction": pred, "probability": round(proba, 3),
                "dropout_risk": "OUI" if pred == 1 else "NON",
                "unit": f"Unite {unit}",
                "message": f"Risque dropout : {'OUI' if pred==1 else 'NON'} (P={round(proba*100,1)}%)"
            })
        else:
            return jsonify({"error": f"Type non supporte. Utilisez 'regression' ou 'classification'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", "message": "ML Scout API running", "version": "1.0.0",
        "endpoints": [
            "POST /predict          → Prediction unifiee (regression/classification)",
            "POST /run/all          → Lance tous les scripts ML",
            "POST /run/mlops        → Lance le pipeline MLOps",
            "GET  /dashboard        → Resultats complets",
            "GET  /report/html      → Rapport HTML enrichi",
            "GET  /health           → Verification serveur"
        ]
    })

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
