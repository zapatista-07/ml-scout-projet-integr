# ML Scout — Monitoring System (S13)

Production-like monitoring with Prometheus + Grafana for the ML Scout MLOps pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML Scout Monitoring Stack                    │
│                                                                  │
│  ┌──────────────┐    /metrics    ┌─────────────┐               │
│  │ Monitoring   │◄──────────────►│ Prometheus  │               │
│  │ API :5001    │                │ :9090       │               │
│  │              │                └──────┬──────┘               │
│  │ • /metrics   │                       │ alerts               │
│  │ • /predict/* │                ┌──────▼──────┐               │
│  │ • /data/*    │                │Alertmanager │               │
│  │ • /simulate/*│                │ :9093       │               │
│  │ • /monitoring│                └──────┬──────┘               │
│  └──────────────┘                       │ webhook              │
│                                  ┌──────▼──────┐               │
│  ┌──────────────┐                │   Grafana   │               │
│  │ ML API :5000 │                │ :3000       │               │
│  │ (original)   │                │ Dashboard   │               │
│  └──────────────┘                └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
cd ml_scout_project
docker-compose up -d
```

Services:
| Service | URL | Credentials |
|---|---|---|
| Grafana Dashboard | http://localhost:3000 | admin / mlscout2024 |
| Prometheus | http://localhost:9090 | — |
| Alertmanager | http://localhost:9093 | — |
| Monitoring API | http://localhost:5001 | — |
| ML API | http://localhost:5000 | — |
| MLflow UI | http://localhost:5002 | — |

### Option B — Local (without Docker)

```bash
cd ml_scout_project
pip install -r requirements_monitoring.txt
python monitoring_api.py
```

Then run simulations:
```bash
python simulate_scenarios.py
```

---

## Monitoring Features

### 1. Prometheus Metrics (`/metrics`)

| Metric | Type | Description |
|---|---|---|
| `ml_scout_requests_total` | Counter | Total API requests by endpoint/status |
| `ml_scout_request_latency_seconds` | Histogram | Request latency (p50/p95/p99) |
| `ml_scout_errors_total` | Counter | Errors by endpoint/type |
| `ml_scout_model_accuracy` | Gauge | Current model accuracy |
| `ml_scout_model_confidence` | Gauge | Prediction confidence |
| `ml_scout_accuracy_delta_from_baseline` | Gauge | Accuracy vs baseline |
| `ml_scout_accuracy_degradation` | Gauge | 1 if accuracy dropped >5% |
| `ml_scout_drift_score` | Gauge | Distribution drift score (0–1) |
| `ml_scout_drift_detected` | Gauge | 1 if drift detected |
| `ml_scout_missing_values_ratio` | Gauge | Missing values per feature |
| `ml_scout_data_freshness_seconds` | Gauge | Seconds since last data update |
| `ml_scout_retraining_triggers_total` | Counter | Retraining trigger count |

Scraping: every **10 seconds**

### 2. Grafana Dashboard

Panels organized in 6 sections:
- **Traffic** — Request rate by endpoint, active requests
- **Performance** — Latency p50/p95/p99 per endpoint
- **Stability** — Error rate over time, global error gauge
- **Model Health** — Accuracy vs baseline, confidence, degradation flags
- **Data Health** — Missing values ratio, data freshness, prediction volume
- **Drift Detection** — Drift score per feature, drift flags, retraining triggers

Auto-refresh: **10 seconds**

### 3. Drift & Degradation Detection

Rules:
- **Data drift**: PSI-like score > 0.3 → warning, > 0.6 → critical
- **Accuracy drop**: > 5% below baseline → critical + retraining trigger
- **Confidence drop**: > 10% below baseline → warning

Baselines:
| Model | Accuracy | Confidence |
|---|---|---|
| Regression (Ridge) | R²=0.770 | 0.85 |
| Classification (LR) | F1=0.800 | 0.80 |
| Clustering (K-Means) | Silhouette=0.6276 | 0.75 |
| Time Series (XGBoost) | R²=0.685 | 0.78 |

### 4. Alerting Rules

| Alert | Condition | Severity |
|---|---|---|
| HighLatency | p95 > 1s | warning |
| CriticalLatency | p95 > 2.5s | critical |
| HighErrorRate | error rate > 10% | warning |
| CriticalErrorRate | error rate > 30% | critical |
| ModelAccuracyDegradation | accuracy drop > 5% | critical |
| DataDriftDetected | drift score > 0.3 | warning |
| SevereDriftScore | drift score > 0.6 | critical |
| HighMissingValues | missing > 10% | warning |
| RetrainingRequired | trigger fired | critical |
| APIDown | service unreachable | critical |

---

## Simulation Scenarios

### Run all scenarios
```bash
python simulate_scenarios.py
```

### Scenario 1 — High Traffic
```bash
python simulate_scenarios.py --scenario traffic --duration 60
```
Sends ~10 req/s → observe latency increase in Grafana

### Scenario 2 — API Errors
```bash
python simulate_scenarios.py --scenario errors --duration 60
```
~40% of requests fail → observe error rate spike

### Scenario 3 — Model Drift
```bash
python simulate_scenarios.py --scenario drift --duration 90
```
Accuracy drops >5%, confidence decreases, distribution shifts → observe degradation alerts and retraining triggers

### Manual scenario activation via API
```bash
# Activate high traffic
curl -X POST http://localhost:5001/simulate/high_traffic

# Activate API errors
curl -X POST http://localhost:5001/simulate/api_errors

# Activate model drift
curl -X POST http://localhost:5001/simulate/model_drift

# Reset to normal
curl -X POST http://localhost:5001/simulate/normal

# Check current status
curl http://localhost:5001/monitoring/status
```

---

## Observability

### Logs

| Log file | Content |
|---|---|
| `logs/monitoring.log` | All API events, errors, anomalies, retraining triggers |
| `logs/simulation.log` | Simulation scenario events |

Log levels:
- `INFO` — Normal predictions, data ingestion
- `WARNING` — Drift detected, confidence drop, high missing values
- `ERROR` — API errors, failed requests
- `CRITICAL` — Accuracy degradation, retraining triggers, severe drift

### Metrics vs Logs

> **Metrics = what happened** (counters, gauges, histograms in Prometheus)
> **Logs = why it happened** (structured log entries with context)

---

## File Structure

```
ml_scout_project/
├── monitoring_api.py          ← Monitoring API (Prometheus + drift + alerts)
├── simulate_scenarios.py      ← 3 mandatory simulation scenarios
├── prometheus.yml             ← Prometheus scrape config (10s interval)
├── alert_rules.yml            ← Alerting rules (latency, errors, drift, accuracy)
├── alertmanager.yml           ← Alertmanager routing + webhook
├── Dockerfile.monitoring      ← Docker image for monitoring API
├── docker-compose.yml         ← Full stack (API + Prometheus + Grafana + AM)
├── requirements_monitoring.txt
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml ← Auto-configure Prometheus datasource
│       └── dashboards/
│           ├── dashboard.yml  ← Dashboard provider config
│           └── ml_scout_dashboard.json ← Full Grafana dashboard
└── logs/
    ├── monitoring.log
    └── simulation.log
```
