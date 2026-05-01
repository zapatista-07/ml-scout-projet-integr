# =============================================================================
# monitoring_api.py — ML Scout Monitoring API (S13)
# Prometheus metrics + drift detection + alerting + simulation scenarios
# =============================================================================
import time
import random
import logging
import threading
import os
import json
from datetime import datetime
from flask import Flask, jsonify, request, Response
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'monitoring.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ml_scout_monitoring')

# ─────────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
registry = CollectorRegistry()

# --- Traffic ---
REQUEST_COUNT = Counter(
    'ml_scout_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

# --- Latency ---
REQUEST_LATENCY = Histogram(
    'ml_scout_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=registry
)

# --- Error rate ---
ERROR_COUNT = Counter(
    'ml_scout_errors_total',
    'Total number of errors',
    ['endpoint', 'error_type'],
    registry=registry
)

# --- Model health ---
MODEL_ACCURACY = Gauge(
    'ml_scout_model_accuracy',
    'Current model accuracy vs baseline',
    ['model_name'],
    registry=registry
)

MODEL_CONFIDENCE = Gauge(
    'ml_scout_model_confidence',
    'Average prediction confidence score',
    ['model_name'],
    registry=registry
)

ACCURACY_BASELINE_DELTA = Gauge(
    'ml_scout_accuracy_delta_from_baseline',
    'Accuracy deviation from baseline (negative = degradation)',
    ['model_name'],
    registry=registry
)

# --- Data health ---
MISSING_VALUES_RATIO = Gauge(
    'ml_scout_missing_values_ratio',
    'Ratio of missing values in incoming data',
    ['feature'],
    registry=registry
)

DATA_FRESHNESS_SECONDS = Gauge(
    'ml_scout_data_freshness_seconds',
    'Seconds since last data update',
    registry=registry
)

PREDICTION_COUNT = Counter(
    'ml_scout_predictions_total',
    'Total predictions made',
    ['model_name', 'prediction_label'],
    registry=registry
)

# --- Drift detection ---
DRIFT_SCORE = Gauge(
    'ml_scout_drift_score',
    'Data drift score (0=no drift, 1=full drift)',
    ['feature'],
    registry=registry
)

DRIFT_DETECTED = Gauge(
    'ml_scout_drift_detected',
    '1 if drift detected, 0 otherwise',
    ['feature'],
    registry=registry
)

ACCURACY_DEGRADATION = Gauge(
    'ml_scout_accuracy_degradation',
    '1 if accuracy degradation detected (>5%), 0 otherwise',
    ['model_name'],
    registry=registry
)

CONFIDENCE_DEGRADATION = Gauge(
    'ml_scout_confidence_degradation',
    '1 if confidence degradation detected, 0 otherwise',
    ['model_name'],
    registry=registry
)

# --- Retraining ---
RETRAINING_TRIGGER = Counter(
    'ml_scout_retraining_triggers_total',
    'Number of retraining triggers fired',
    ['reason'],
    registry=registry
)

# --- Active requests ---
ACTIVE_REQUESTS = Gauge(
    'ml_scout_active_requests',
    'Number of currently active requests',
    registry=registry
)

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE VALUES (production reference)
# ─────────────────────────────────────────────────────────────────────────────
BASELINES = {
    'regression': {
        'accuracy': 0.770,       # R²
        'confidence': 0.85,
        'latency_p95': 0.5,      # seconds
    },
    'classification': {
        'accuracy': 0.800,       # F1
        'confidence': 0.80,
        'latency_p95': 0.3,
    },
    'clustering': {
        'accuracy': 0.6276,      # Silhouette
        'confidence': 0.75,
        'latency_p95': 0.4,
    },
    'timeseries': {
        'accuracy': 0.685,       # R²
        'confidence': 0.78,
        'latency_p95': 0.6,
    }
}

# Initialize baseline metrics
for model, vals in BASELINES.items():
    MODEL_ACCURACY.labels(model_name=model).set(vals['accuracy'])
    MODEL_CONFIDENCE.labels(model_name=model).set(vals['confidence'])
    ACCURACY_BASELINE_DELTA.labels(model_name=model).set(0.0)
    ACCURACY_DEGRADATION.labels(model_name=model).set(0)
    CONFIDENCE_DEGRADATION.labels(model_name=model).set(0)

# Initialize drift metrics
for feature in ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1', 'retention_rate']:
    DRIFT_SCORE.labels(feature=feature).set(0.0)
    DRIFT_DETECTED.labels(feature=feature).set(0)

# Initialize data health
for feature in ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1']:
    MISSING_VALUES_RATIO.labels(feature=feature).set(0.0)
DATA_FRESHNESS_SECONDS.set(0)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION STATE
# ─────────────────────────────────────────────────────────────────────────────
simulation_state = {
    'high_traffic': False,
    'api_errors': False,
    'model_drift': False,
    'normal': True
}

# ─────────────────────────────────────────────────────────────────────────────
# DRIFT DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class DriftDetector:
    """Simple threshold-based drift and degradation detector."""

    DRIFT_THRESHOLD = 0.3       # drift score above this → alert
    ACCURACY_DROP_THRESHOLD = 0.05   # >5% drop → alert
    CONFIDENCE_DROP_THRESHOLD = 0.10  # >10% drop → alert

    def __init__(self):
        self.reference_distributions = {
            'members_lag1':     {'mean': 13.8, 'std': 3.2},
            'leak_rate_lag1':   {'mean': 48.5, 'std': 12.0},
            'growth_rate_lag1': {'mean': 2.1,  'std': 8.5},
            'retention_rate':   {'mean': 0.515, 'std': 0.12},
        }
        self.current_accuracies = {m: v['accuracy'] for m, v in BASELINES.items()}
        self.current_confidences = {m: v['confidence'] for m, v in BASELINES.items()}

    def compute_drift_score(self, feature, current_mean, current_std):
        """PSI-like drift score based on mean shift."""
        ref = self.reference_distributions.get(feature)
        if not ref:
            return 0.0
        mean_shift = abs(current_mean - ref['mean']) / (ref['std'] + 1e-9)
        std_shift  = abs(current_std  - ref['std'])  / (ref['std'] + 1e-9)
        score = min(1.0, (mean_shift * 0.6 + std_shift * 0.4) / 3.0)
        return round(score, 4)

    def check_drift(self, feature, current_mean, current_std):
        score = self.compute_drift_score(feature, current_mean, current_std)
        DRIFT_SCORE.labels(feature=feature).set(score)
        detected = 1 if score > self.DRIFT_THRESHOLD else 0
        DRIFT_DETECTED.labels(feature=feature).set(detected)

        if detected:
            logger.warning(
                f"[DRIFT] Feature '{feature}' drift detected! "
                f"score={score:.3f} (threshold={self.DRIFT_THRESHOLD}) "
                f"current_mean={current_mean:.2f} ref_mean={self.reference_distributions[feature]['mean']:.2f}"
            )
            self._check_retraining_trigger('drift')
        return score, detected

    def check_accuracy_degradation(self, model_name, current_accuracy):
        baseline = BASELINES[model_name]['accuracy']
        delta = current_accuracy - baseline
        ACCURACY_BASELINE_DELTA.labels(model_name=model_name).set(round(delta, 4))
        MODEL_ACCURACY.labels(model_name=model_name).set(round(current_accuracy, 4))
        self.current_accuracies[model_name] = current_accuracy

        degraded = 1 if delta < -self.ACCURACY_DROP_THRESHOLD else 0
        ACCURACY_DEGRADATION.labels(model_name=model_name).set(degraded)

        if degraded:
            logger.error(
                f"[DEGRADATION] Model '{model_name}' accuracy dropped! "
                f"current={current_accuracy:.3f} baseline={baseline:.3f} "
                f"delta={delta:.3f} (threshold=-{self.ACCURACY_DROP_THRESHOLD})"
            )
            self._check_retraining_trigger('accuracy_drop')
        return delta, degraded

    def check_confidence_degradation(self, model_name, current_confidence):
        baseline = BASELINES[model_name]['confidence']
        delta = current_confidence - baseline
        MODEL_CONFIDENCE.labels(model_name=model_name).set(round(current_confidence, 4))
        self.current_confidences[model_name] = current_confidence

        degraded = 1 if delta < -self.CONFIDENCE_DROP_THRESHOLD else 0
        CONFIDENCE_DEGRADATION.labels(model_name=model_name).set(degraded)

        if degraded:
            logger.warning(
                f"[CONFIDENCE] Model '{model_name}' confidence dropped! "
                f"current={current_confidence:.3f} baseline={baseline:.3f} "
                f"delta={delta:.3f}"
            )
        return delta, degraded

    def _check_retraining_trigger(self, reason):
        RETRAINING_TRIGGER.labels(reason=reason).inc()
        logger.critical(
            f"[RETRAINING TRIGGER] Reason: {reason} — "
            f"Automatic retraining should be initiated!"
        )


drift_detector = DriftDetector()

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


def track_request(endpoint, method='GET'):
    """Context manager-like decorator for tracking requests."""
    def decorator(f):
        def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start = time.time()
            status = '200'
            try:
                # Simulate latency under high traffic
                if simulation_state['high_traffic']:
                    time.sleep(random.uniform(0.5, 2.0))
                else:
                    time.sleep(random.uniform(0.01, 0.1))

                # Simulate API errors
                if simulation_state['api_errors'] and random.random() < 0.4:
                    raise RuntimeError("Simulated API error")

                result = f(*args, **kwargs)
                return result
            except Exception as e:
                status = '500'
                ERROR_COUNT.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
                logger.error(f"[ERROR] {endpoint} — {type(e).__name__}: {e}")
                return jsonify({'error': str(e), 'endpoint': endpoint}), 500
            finally:
                latency = time.time() - start
                REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
                ACTIVE_REQUESTS.dec()
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(registry), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'ML Scout Monitoring API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'simulation': simulation_state
    })


@app.route('/predict/regression', methods=['POST'])
@track_request('/predict/regression', 'POST')
def predict_regression():
    data = request.get_json() or {}
    unit = int(data.get('fk_type_unite', random.randint(1, 6)))

    # Simulate drift scenario
    if simulation_state['model_drift']:
        confidence = random.uniform(0.45, 0.65)
        accuracy   = BASELINES['regression']['accuracy'] - random.uniform(0.06, 0.15)
    else:
        confidence = random.uniform(0.78, 0.95)
        accuracy   = BASELINES['regression']['accuracy'] + random.uniform(-0.02, 0.02)

    prediction = random.uniform(8, 20)
    drift_detector.check_accuracy_degradation('regression', accuracy)
    drift_detector.check_confidence_degradation('regression', confidence)
    PREDICTION_COUNT.labels(model_name='regression', prediction_label='members').inc()

    logger.info(f"[PREDICT] regression unit={unit} pred={prediction:.1f} conf={confidence:.3f}")
    return jsonify({
        'model': 'Ridge Regression',
        'unit': unit,
        'prediction': round(prediction, 1),
        'confidence': round(confidence, 3),
        'accuracy': round(accuracy, 3),
        'season': '2026/2027'
    })


@app.route('/predict/classification', methods=['POST'])
@track_request('/predict/classification', 'POST')
def predict_classification():
    data = request.get_json() or {}
    unit = int(data.get('fk_type_unite', random.randint(1, 6)))

    if simulation_state['model_drift']:
        confidence = random.uniform(0.40, 0.60)
        accuracy   = BASELINES['classification']['accuracy'] - random.uniform(0.07, 0.18)
    else:
        confidence = random.uniform(0.72, 0.92)
        accuracy   = BASELINES['classification']['accuracy'] + random.uniform(-0.02, 0.02)

    pred_label = 'dropout' if random.random() < 0.3 else 'no_dropout'
    drift_detector.check_accuracy_degradation('classification', accuracy)
    drift_detector.check_confidence_degradation('classification', confidence)
    PREDICTION_COUNT.labels(model_name='classification', prediction_label=pred_label).inc()

    logger.info(f"[PREDICT] classification unit={unit} label={pred_label} conf={confidence:.3f}")
    return jsonify({
        'model': 'Logistic Regression',
        'unit': unit,
        'prediction': pred_label,
        'confidence': round(confidence, 3),
        'accuracy': round(accuracy, 3)
    })


@app.route('/predict/anomaly', methods=['POST'])
@track_request('/predict/anomaly', 'POST')
def predict_anomaly():
    data = request.get_json() or {}
    unit = int(data.get('fk_type_unite', random.randint(1, 6)))

    is_anomaly = random.random() < (0.4 if simulation_state['model_drift'] else 0.15)
    label = 'anomaly' if is_anomaly else 'normal'
    PREDICTION_COUNT.labels(model_name='anomaly', prediction_label=label).inc()

    if is_anomaly:
        logger.warning(f"[ANOMALY] Anomaly detected for unit={unit}")

    return jsonify({'unit': unit, 'anomaly': is_anomaly, 'label': label})


@app.route('/data/ingest', methods=['POST'])
@track_request('/data/ingest', 'POST')
def data_ingest():
    """Simulates data ingestion with drift and missing value checks."""
    data = request.get_json() or {}

    # Simulate missing values
    features = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1']
    missing_ratios = {}
    for feat in features:
        ratio = random.uniform(0.0, 0.15) if simulation_state['model_drift'] else random.uniform(0.0, 0.03)
        MISSING_VALUES_RATIO.labels(feature=feat).set(round(ratio, 4))
        missing_ratios[feat] = round(ratio, 4)
        if ratio > 0.1:
            logger.warning(f"[DATA] High missing values for '{feat}': {ratio:.1%}")

    # Simulate drift
    drift_results = {}
    if simulation_state['model_drift']:
        distributions = {
            'members_lag1':     (18.5, 6.0),
            'leak_rate_lag1':   (65.0, 18.0),
            'growth_rate_lag1': (-5.0, 12.0),
            'retention_rate':   (0.35, 0.18),
        }
    else:
        distributions = {
            'members_lag1':     (13.8 + random.uniform(-1, 1), 3.2),
            'leak_rate_lag1':   (48.5 + random.uniform(-3, 3), 12.0),
            'growth_rate_lag1': (2.1  + random.uniform(-2, 2), 8.5),
            'retention_rate':   (0.515 + random.uniform(-0.02, 0.02), 0.12),
        }

    for feat, (mean, std) in distributions.items():
        score, detected = drift_detector.check_drift(feat, mean, std)
        drift_results[feat] = {'score': score, 'drift_detected': bool(detected)}

    DATA_FRESHNESS_SECONDS.set(random.uniform(0, 300))

    return jsonify({
        'status': 'ingested',
        'missing_values': missing_ratios,
        'drift_results': drift_results,
        'timestamp': datetime.utcnow().isoformat()
    })


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/simulate/high_traffic', methods=['POST'])
def simulate_high_traffic():
    """Scenario 1: High traffic → observe latency impact."""
    simulation_state.update({'high_traffic': True, 'api_errors': False, 'model_drift': False, 'normal': False})
    logger.warning("[SIMULATION] HIGH TRAFFIC scenario activated — expect latency increase")
    return jsonify({'scenario': 'high_traffic', 'status': 'activated',
                    'description': 'High traffic simulation: latency will increase significantly'})


@app.route('/simulate/api_errors', methods=['POST'])
def simulate_api_errors():
    """Scenario 2: API errors → observe error spikes."""
    simulation_state.update({'high_traffic': False, 'api_errors': True, 'model_drift': False, 'normal': False})
    logger.error("[SIMULATION] API ERRORS scenario activated — expect error rate spike")
    return jsonify({'scenario': 'api_errors', 'status': 'activated',
                    'description': 'API errors simulation: ~40% of requests will fail'})


@app.route('/simulate/model_drift', methods=['POST'])
def simulate_model_drift():
    """Scenario 3: Model drift → observe performance degradation."""
    simulation_state.update({'high_traffic': False, 'api_errors': False, 'model_drift': True, 'normal': False})
    logger.critical("[SIMULATION] MODEL DRIFT scenario activated — accuracy and confidence will degrade")
    return jsonify({'scenario': 'model_drift', 'status': 'activated',
                    'description': 'Model drift simulation: accuracy drop >5%, confidence decrease, data distribution shift'})


@app.route('/simulate/normal', methods=['POST'])
def simulate_normal():
    """Reset to normal operation."""
    simulation_state.update({'high_traffic': False, 'api_errors': False, 'model_drift': False, 'normal': True})
    # Reset drift metrics
    for feature in ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1', 'retention_rate']:
        DRIFT_SCORE.labels(feature=feature).set(0.0)
        DRIFT_DETECTED.labels(feature=feature).set(0)
    for model in BASELINES:
        MODEL_ACCURACY.labels(model_name=model).set(BASELINES[model]['accuracy'])
        MODEL_CONFIDENCE.labels(model_name=model).set(BASELINES[model]['confidence'])
        ACCURACY_BASELINE_DELTA.labels(model_name=model).set(0.0)
        ACCURACY_DEGRADATION.labels(model_name=model).set(0)
        CONFIDENCE_DEGRADATION.labels(model_name=model).set(0)
    logger.info("[SIMULATION] Normal operation restored")
    return jsonify({'scenario': 'normal', 'status': 'activated',
                    'description': 'All metrics reset to baseline values'})


@app.route('/simulate/status', methods=['GET'])
def simulate_status():
    return jsonify({'simulation_state': simulation_state})


# ─────────────────────────────────────────────────────────────────────────────
# ALERTMANAGER WEBHOOK RECEIVER
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/alerts/webhook', methods=['POST'])
def alerts_webhook():
    """Receives alerts from Alertmanager and logs them."""
    payload = request.get_json() or {}
    alerts = payload.get('alerts', [])

    for alert in alerts:
        name     = alert.get('labels', {}).get('alertname', 'Unknown')
        severity = alert.get('labels', {}).get('severity', 'unknown')
        status   = alert.get('status', 'unknown')
        summary  = alert.get('annotations', {}).get('summary', '')
        desc     = alert.get('annotations', {}).get('description', '')

        if status == 'firing':
            if severity == 'critical':
                logger.critical(f"[ALERT FIRING] [{severity.upper()}] {name}: {summary} — {desc}")
            else:
                logger.warning(f"[ALERT FIRING] [{severity.upper()}] {name}: {summary} — {desc}")
        elif status == 'resolved':
            logger.info(f"[ALERT RESOLVED] {name}: {summary}")

    return jsonify({'received': len(alerts), 'status': 'logged'}), 200


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT & DEGRADATION STATUS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/monitoring/status', methods=['GET'])
def monitoring_status():
    """Full monitoring status — metrics, drift, degradation."""
    status = {
        'timestamp': datetime.utcnow().isoformat(),
        'simulation': simulation_state,
        'baselines': BASELINES,
        'models': {},
        'drift': {},
        'alerts': []
    }

    for model, vals in BASELINES.items():
        curr_acc  = MODEL_ACCURACY.labels(model_name=model)._value.get()
        curr_conf = MODEL_CONFIDENCE.labels(model_name=model)._value.get()
        delta     = ACCURACY_BASELINE_DELTA.labels(model_name=model)._value.get()
        degraded  = ACCURACY_DEGRADATION.labels(model_name=model)._value.get()
        conf_deg  = CONFIDENCE_DEGRADATION.labels(model_name=model)._value.get()

        status['models'][model] = {
            'current_accuracy':  round(curr_acc, 4),
            'baseline_accuracy': vals['accuracy'],
            'delta':             round(delta, 4),
            'accuracy_degraded': bool(degraded),
            'current_confidence': round(curr_conf, 4),
            'confidence_degraded': bool(conf_deg)
        }

        if degraded:
            status['alerts'].append({
                'type': 'ACCURACY_DEGRADATION',
                'model': model,
                'message': f"Accuracy dropped {abs(delta):.1%} below baseline",
                'severity': 'critical'
            })
        if conf_deg:
            status['alerts'].append({
                'type': 'CONFIDENCE_DEGRADATION',
                'model': model,
                'message': f"Confidence below threshold",
                'severity': 'warning'
            })

    for feature in ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1', 'retention_rate']:
        score    = DRIFT_SCORE.labels(feature=feature)._value.get()
        detected = DRIFT_DETECTED.labels(feature=feature)._value.get()
        status['drift'][feature] = {
            'score': round(score, 4),
            'detected': bool(detected)
        }
        if detected:
            status['alerts'].append({
                'type': 'DATA_DRIFT',
                'feature': feature,
                'message': f"Distribution drift detected (score={score:.3f})",
                'severity': 'warning'
            })

    return jsonify(status)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TRAFFIC GENERATOR (for realistic metrics)
# ─────────────────────────────────────────────────────────────────────────────

def background_traffic_generator():
    """Generates background traffic to keep metrics alive."""
    import urllib.request
    base_url = 'http://localhost:5001'
    endpoints = [
        ('/predict/regression', 'POST', '{"fk_type_unite": 1}'),
        ('/predict/classification', 'POST', '{"fk_type_unite": 2}'),
        ('/predict/anomaly', 'POST', '{"fk_type_unite": 3}'),
        ('/data/ingest', 'POST', '{}'),
    ]
    while True:
        try:
            ep, method, body = random.choice(endpoints)
            req = urllib.request.Request(
                base_url + ep,
                data=body.encode(),
                headers={'Content-Type': 'application/json'},
                method=method
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        interval = random.uniform(1, 3) if simulation_state['high_traffic'] else random.uniform(3, 8)
        time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start background traffic generator
    t = threading.Thread(target=background_traffic_generator, daemon=True)
    t.start()
    logger.info("ML Scout Monitoring API starting on port 5001")
    logger.info("Prometheus metrics available at http://localhost:5001/metrics")
    app.run(host='0.0.0.0', port=5001, debug=False)
