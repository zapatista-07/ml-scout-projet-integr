# 🏆 ML Scout — Projet Intégré Machine Learning

Prédiction du nombre d'adhérents par unité et par saison pour un groupe scout.

## 📊 Sections ML

| Section | Modèle | Métrique |
|---|---|---|
| D — Régression | Ridge Regression | R²=0.770 |
| C — Classification | Logistic Regression | F1=0.800 |
| E — Clustering | K-Means + GMM | Silhouette=0.6276 |
| F — Time Series | SARIMA + XGBoost TS | R²=0.685 |
| Advanced — Anomaly Detection | Isolation Forest + LOF + Z-Score | 2 anomalies |

---

## 🆕 S13 — Monitoring MLOps (Prometheus + Grafana)

Système de monitoring production-like pour surveiller l'API, les modèles et les données en temps réel.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  ML Scout — Stack Complète               │
│                                                         │
│  Flask ML API :5000   ←→   Monitoring API :5001         │
│                                  ↓                      │
│                          Prometheus :9090               │
│                          (scrape /metrics toutes 10s)   │
│                                  ↓                      │
│                    ┌─────────────┴──────────┐           │
│                    ↓                        ↓           │
│             Grafana :3000          Alertmanager :9093   │
│             (dashboards)           (alertes critiques)  │
│                                                         │
│  MLflow UI :5002  (tracking expériences + registry)     │
└─────────────────────────────────────────────────────────┘
```

### Services Docker

| Service | URL | Credentials |
|---|---|---|
| ML API (Flask) | http://localhost:5000 | — |
| Swagger UI | http://localhost:5000/apidocs | — |
| Monitoring API | http://localhost:5001 | — |
| Grafana | http://localhost:3000 | admin / mlscout2024 |
| Prometheus | http://localhost:9090 | — |
| Alertmanager | http://localhost:9093 | — |
| MLflow UI | http://localhost:5002 | — |

### Métriques Prometheus

| Métrique | Type | Description |
|---|---|---|
| `ml_scout_requests_total` | Counter | Requêtes totales par endpoint/status |
| `ml_scout_request_latency_seconds` | Histogram | Latence p50/p95/p99 |
| `ml_scout_errors_total` | Counter | Erreurs par endpoint |
| `ml_scout_model_accuracy` | Gauge | Précision du modèle en cours |
| `ml_scout_model_confidence` | Gauge | Confiance des prédictions |
| `ml_scout_drift_score` | Gauge | Score de dérive (0–1) |
| `ml_scout_drift_detected` | Gauge | 1 si drift détecté |
| `ml_scout_accuracy_degradation` | Gauge | 1 si précision chutée > 5% |
| `ml_scout_retraining_triggers_total` | Counter | Déclenchements de réentraînement |

### Grafana Dashboard — 6 sections

- **Traffic** — Taux de requêtes par endpoint (req/s)
- **Performance** — Latence p50/p95/p99
- **Stability** — Taux d'erreurs
- **Model Health** — Précision vs baseline, confiance, dégradation
- **Data Health** — Valeurs manquantes, fraîcheur des données
- **Drift Detection** — Score de dérive par feature, triggers de réentraînement

### Alertes configurées

| Alerte | Condition | Sévérité |
|---|---|---|
| HighLatency | p95 > 1s | warning |
| CriticalLatency | p95 > 2.5s | critical |
| HighErrorRate | taux erreurs > 10% | warning |
| ModelAccuracyDegradation | précision chute > 5% | critical |
| DataDriftDetected | drift score > 0.3 | warning |
| SevereDriftScore | drift score > 0.6 | critical |
| RetrainingRequired | trigger déclenché | critical |

### Détection Drift & Dégradation

- **Data drift** : score PSI > 0.3 → warning, > 0.6 → critical
- **Accuracy drop** : > 5% sous le baseline → alerte critique + trigger réentraînement
- **Confidence drop** : > 10% sous le baseline → warning

Baselines :
| Modèle | Accuracy | Confidence |
|---|---|---|
| Ridge Regression | R²=0.770 | 0.85 |
| Logistic Regression | F1=0.800 | 0.80 |
| K-Means | Silhouette=0.6276 | 0.75 |
| XGBoost TS | R²=0.685 | 0.78 |

### Scénarios de simulation

```bash
# Lancer tous les scénarios (3 × 60s)
python simulate_scenarios.py

# Scénario 1 — High Traffic (~10 req/s pendant 60s)
python simulate_scenarios.py --scenario traffic --duration 60

# Scénario 2 — API Errors (~40% d'erreurs pendant 60s)
python simulate_scenarios.py --scenario errors --duration 60

# Scénario 3 — Model Drift (accuracy drop >5%, drift détecté)
python simulate_scenarios.py --scenario drift --duration 90
```

### Swagger UI

Documentation interactive de l'API disponible sur `http://localhost:5000/apidocs`

Endpoints documentés :
- `GET  /health` — Status du serveur
- `POST /predict` — Prédiction unifiée (régression ou classification)
- `GET  /dashboard` — Résultats complets
- `GET  /results/regression` — Forecasts régression
- `GET  /results/classification` — Prédictions dropout
- `GET  /results/anomaly` — Anomalies détectées
- `GET  /results/timeseries` — Forecasts time series
- `POST /run/all` — Lance tous les scripts ML
- `POST /run/mlops` — Lance le pipeline MLOps
- `GET  /report/html` — Rapport HTML complet

---

## 🗂️ Structure du projet

```
ml_scout_project/
├── data/                          ← Datasets DWH
├── models/                        ← Modèles sauvegardés + résultats CSV
├── visuals/                       ← Graphiques générés
├── logs/
│   ├── monitoring.log             ← Logs API (erreurs, anomalies, triggers)
│   └── simulation.log             ← Logs scénarios de simulation
├── grafana/
│   └── provisioning/
│       ├── datasources/           ← Config datasource Prometheus
│       └── dashboards/            ← Dashboard JSON Grafana
├── 01_data_preparation.py
├── 02_regression.py
├── 03_classification.py
├── 04_clustering.py
├── 06_anomaly_detection.py
├── 07_mlops_mlflow.py             ← MLflow tracking
├── 08_mlops_registry.py           ← Model registry
├── 09_mlops_staging_production.py ← Staging / Production
├── flask_api.py                   ← API Flask + Swagger UI
├── monitoring_api.py              ← API Monitoring (métriques Prometheus)
├── simulate_scenarios.py          ← Scénarios de simulation S13
├── prometheus.yml                 ← Config scraping Prometheus
├── alert_rules.yml                ← Règles d'alertes
├── alertmanager.yml               ← Config Alertmanager
├── docker-compose.yml             ← Stack complète Docker
├── Dockerfile                     ← Image ML API
├── Dockerfile.monitoring          ← Image Monitoring API
├── requirements.txt               ← Dépendances ML API
├── requirements_monitoring.txt    ← Dépendances Monitoring API
├── app.py                         ← Site web Flask
└── build_features.py              ← Feature engineering
05_timeseries.py                   ← Time Series (dossier racine)
```

---

## 🚀 Lancement

### Avec Docker (recommandé)

```bash
cd ml_scout_project
docker-compose up -d
```

Puis ouvrir http://localhost:3000 (Grafana) et lancer une simulation :

```bash
python simulate_scenarios.py
```

### Sans Docker (local)

```bash
cd ml_scout_project
pip install -r requirements.txt
python flask_api.py
# → http://localhost:5000
# → http://localhost:5000/apidocs (Swagger)

pip install -r requirements_monitoring.txt
python monitoring_api.py
# → http://localhost:5001
```

---

## 🤖 Automatisation n8n

Pipeline automatique mensuel via n8n + Docker :
- Lance les 5 scripts ML
- Récupère les résultats via l'API
- Envoie un email d'alerte si anomalie détectée

---

## 📈 Résultats

- **18 observations DWH** (6 unités × 3 saisons)
- **Validation LOO-CV** (Leave-One-Out)
- **Features laggées** : members_lag1, leak_rate_lag1, growth_rate_lag1, etc.
- **2 anomalies confirmées** : U1 (2025/2026) et U2 (2024/2025)

---

## 👨‍💻 Technologies

Python · Flask · scikit-learn · statsmodels · pandas · matplotlib · MLflow · Prometheus · Grafana · Alertmanager · Docker · n8n · Swagger (Flasgger)
