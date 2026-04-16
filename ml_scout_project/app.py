# =============================================================================
# app.py — Flask Web Application
# ML Scout Project — Prédiction des adhérents
# =============================================================================
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
MODELS  = os.path.join(BASE, 'models')
DATA    = os.path.join(BASE, 'data')

# ── Chargement des modèles ────────────────────────────────────────────────────
ridge   = joblib.load(os.path.join(MODELS, 'regressor_ridge.pkl'))
lr_cls  = joblib.load(os.path.join(MODELS, 'classifier_lr.pkl'))
kmeans  = joblib.load(os.path.join(MODELS, 'cluster_kmeans.pkl'))
scaler_clu = joblib.load(os.path.join(MODELS, 'clustering_scaler.pkl'))

# ── Features ──────────────────────────────────────────────────────────────────
FEAT_REG = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
            'delta_members_lag1', 'avg_members_unite_past',
            'season_index', 'fk_type_unite',
            'retention_rate', 'members_x_retention']

FEAT_CLS = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
            'delta_members_lag1', 'avg_members_unite_past',
            'season_index', 'fk_type_unite',
            'retention_rate', 'members_x_retention']

FEAT_CLU = ['avg_variation', 'cv_members']

# ── Données DWH ───────────────────────────────────────────────────────────────
df_ts  = pd.read_csv(os.path.join(DATA, 'dataset_timeseries.csv'))
df_clu = pd.read_csv(os.path.join(DATA, 'dataset_clustering.csv'))
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)
df = df_ts.merge(df_clu[['fk_type_unite','season_year','leak_rate',
                          'nmbr_members_previous_season']],
                 on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

# Résultats pré-calculés
fc_reg  = pd.read_csv(os.path.join(MODELS, 'predictions_2026_2027.csv'))
fc_ts   = pd.read_csv(os.path.join(MODELS, 'forecasts_ts_2026_2027.csv'))
fc_drop = pd.read_csv(os.path.join(MODELS, 'predictions_dropout_2026.csv'))
anomaly = pd.read_csv(os.path.join(MODELS, 'anomaly_results.csv'))
reg_cmp = pd.read_csv(os.path.join(MODELS, 'regression_comparison.csv'))
cls_cmp = pd.read_csv(os.path.join(MODELS, 'classification_comparison.csv'))
clu_cmp = pd.read_csv(os.path.join(MODELS, 'clustering_comparison.csv'))
ts_cmp  = pd.read_csv(os.path.join(MODELS, 'timeseries_comparison.csv'))
clu_pro = pd.read_csv(os.path.join(MODELS, 'cluster_profile.csv'))

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    # Données pour le dashboard
    units = sorted(df['fk_type_unite'].unique())
    unit_data = []
    for u in units:
        sub = df[df['fk_type_unite']==u].sort_values('season_year')
        history = sub[['season','nbmr_members_season','leak_rate']].to_dict('records')
        # Forecast
        fc_r = fc_reg[fc_reg['fk_type_unite']==u]['pred_ridge'].values
        fc_t = fc_ts[fc_ts['Unité']==u]['XGBoost_2026/2027'].values
        fc_d = fc_drop[fc_drop['fk_type_unite']==u]
        anom = anomaly[anomaly['fk_type_unite']==u]['consensus'].sum()
        unit_data.append({
            'id': int(u),
            'history': history,
            'forecast_reg': round(float(fc_r[0]),1) if len(fc_r) else '-',
            'forecast_ts':  round(float(fc_t[0]),1) if len(fc_t) else '-',
            'dropout_proba': round(float(fc_d['dropout_proba'].values[0]),3) if len(fc_d) else '-',
            'dropout_pred':  int(fc_d['dropout_risk_pred'].values[0]) if len(fc_d) else 0,
            'anomalies': int(anom),
            'last_members': int(sub['nbmr_members_season'].iloc[-1]),
            'last_leak': float(sub['leak_rate'].iloc[-1]),
        })

    # KPIs globaux
    total_members = int(df[df['season_year']==2025]['nbmr_members_season'].sum())
    total_forecast = round(float(fc_ts['XGBoost_2026/2027'].sum()), 1)
    total_anomalies = int(anomaly['consensus'].sum())
    dropout_units = int((fc_drop['dropout_risk_pred'] == 1).sum())

    return render_template('index.html',
        unit_data=unit_data,
        total_members=total_members,
        total_forecast=total_forecast,
        total_anomalies=total_anomalies,
        dropout_units=dropout_units,
        reg_cmp=reg_cmp.to_dict('records'),
        cls_cmp=cls_cmp.to_dict('records'),
        clu_cmp=clu_cmp.to_dict('records'),
        ts_cmp=ts_cmp.to_dict('records'),
        clu_pro=clu_pro.to_dict('records'),
    )

# ── API : Prédiction régression ───────────────────────────────────────────────
@app.route('/api/predict_regression', methods=['POST'])
def predict_regression():
    data = request.json
    unit = int(data['fk_type_unite'])
    sub  = df[df['fk_type_unite']==unit].sort_values('season_year')
    last = sub.iloc[-1]
    prev = sub.iloc[-2] if len(sub) >= 2 else sub.iloc[-1]

    m_curr = float(data.get('members_lag1', last['nbmr_members_season']))
    leak1  = float(data.get('leak_rate_lag1', last['leak_rate']))
    m_prev = float(prev['nbmr_members_season'])
    growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta  = m_curr - m_prev
    ret    = 1 - leak1 / 100

    row = {
        'members_lag1':           m_curr,
        'leak_rate_lag1':         leak1,
        'growth_rate_lag1':       growth,
        'delta_members_lag1':     delta,
        'avg_members_unite_past': float(sub['nbmr_members_season'].mean()),
        'season_index':           4,
        'fk_type_unite':          unit,
        'retention_rate':         ret,
        'members_x_retention':    m_curr * ret,
    }
    X = pd.DataFrame([row])[FEAT_REG].values
    pred = max(float(ridge.predict(X)[0]), 0)
    return jsonify({'prediction': round(pred, 1), 'unit': unit})

# ── API : Prédiction dropout ──────────────────────────────────────────────────
@app.route('/api/predict_dropout', methods=['POST'])
def predict_dropout():
    data = request.json
    unit = int(data['fk_type_unite'])
    sub  = df[df['fk_type_unite']==unit].sort_values('season_year')
    last = sub.iloc[-1]
    prev = sub.iloc[-2] if len(sub) >= 2 else sub.iloc[-1]

    m_curr = float(data.get('members_lag1', last['nbmr_members_season']))
    leak1  = float(data.get('leak_rate_lag1', last['leak_rate']))
    m_prev = float(prev['nbmr_members_season'])
    growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta  = m_curr - m_prev
    ret    = 1 - leak1 / 100

    row = {
        'members_lag1':           m_curr,
        'leak_rate_lag1':         leak1,
        'growth_rate_lag1':       growth,
        'delta_members_lag1':     delta,
        'avg_members_unite_past': float(sub['nbmr_members_season'].mean()),
        'season_index':           4,
        'fk_type_unite':          unit,
        'retention_rate':         ret,
        'members_x_retention':    m_curr * ret,
    }
    X = pd.DataFrame([row])[FEAT_CLS].values
    pred  = int(lr_cls.predict(X)[0])
    proba = float(lr_cls.predict_proba(X)[0][1])
    return jsonify({'dropout': pred, 'probability': round(proba, 3), 'unit': unit})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
