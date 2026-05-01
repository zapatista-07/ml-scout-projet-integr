#!/usr/bin/env python3
# =============================================================================
# simulate_scenarios.py — ML Scout Monitoring Simulation Scenarios (S13)
#
# Runs 3 mandatory scenarios:
#   1. High Traffic  → observe latency impact
#   2. API Errors    → observe error spikes
#   3. Model Drift   → observe performance degradation
#
# Usage:
#   python simulate_scenarios.py                    # all scenarios
#   python simulate_scenarios.py --scenario traffic
#   python simulate_scenarios.py --scenario errors
#   python simulate_scenarios.py --scenario drift
#   python simulate_scenarios.py --scenario all
# =============================================================================
import time
import json
import random
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('simulation')

BASE_URL = 'http://localhost:5001'


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def post(path, body=None):
    url = BASE_URL + path
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={'Content-Type': 'application/json'},
                                  method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return {'error': e.reason}, e.code
    except Exception as e:
        return {'error': str(e)}, 0


def get(path):
    url = BASE_URL + path
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read()), resp.status
    except Exception as e:
        return {'error': str(e)}, 0


def print_separator(title):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1 — HIGH TRAFFIC
# ─────────────────────────────────────────────────────────────────────────────
def scenario_high_traffic(duration=60, rps=10):
    print_separator('SCENARIO 1: HIGH TRAFFIC → Latency Impact')
    logger.warning('[SCENARIO] Starting HIGH TRAFFIC simulation')

    # Activate scenario
    resp, status = post('/simulate/high_traffic')
    print(f'  Activated: {resp}')

    endpoints = [
        ('/predict/regression',    {'fk_type_unite': 1}),
        ('/predict/classification', {'fk_type_unite': 2}),
        ('/predict/anomaly',        {'fk_type_unite': 3}),
        ('/data/ingest',            {}),
    ]

    print(f'\n  Sending ~{rps} req/s for {duration}s...')
    print('  Watch Grafana → Traffic panel and Latency panel\n')

    start = time.time()
    total_requests = 0
    errors = 0

    while time.time() - start < duration:
        batch_start = time.time()

        for _ in range(rps):
            ep, body = random.choice(endpoints)
            resp, status = post(ep, body)
            total_requests += 1
            if status >= 400 or status == 0:
                errors += 1

        elapsed = time.time() - batch_start
        sleep_time = max(0, 1.0 - elapsed)
        time.sleep(sleep_time)

        remaining = int(duration - (time.time() - start))
        print(f'\r  [{remaining:3d}s left] Requests: {total_requests} | Errors: {errors}', end='', flush=True)

    print(f'\n\n  ✅ High traffic done: {total_requests} requests, {errors} errors')
    logger.info(f'[SCENARIO] High traffic complete: {total_requests} requests, {errors} errors')

    # Reset
    post('/simulate/normal')
    logger.info('[SCENARIO] Returned to normal after high traffic')


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2 — API ERRORS
# ─────────────────────────────────────────────────────────────────────────────
def scenario_api_errors(duration=60):
    print_separator('SCENARIO 2: API ERRORS → Error Rate Spike')
    logger.error('[SCENARIO] Starting API ERRORS simulation')

    resp, _ = post('/simulate/api_errors')
    print(f'  Activated: {resp}')
    print(f'\n  Sending requests for {duration}s with ~40% error rate...')
    print('  Watch Grafana → Error Rate panel\n')

    start = time.time()
    total = 0
    errors = 0

    while time.time() - start < duration:
        ep = random.choice(['/predict/regression', '/predict/classification', '/predict/anomaly'])
        resp, status = post(ep, {'fk_type_unite': random.randint(1, 6)})
        total += 1
        if status >= 400 or status == 0:
            errors += 1
            logger.error(f'[SCENARIO] Error on {ep}: status={status}')

        time.sleep(random.uniform(0.1, 0.5))
        remaining = int(duration - (time.time() - start))
        print(f'\r  [{remaining:3d}s left] Requests: {total} | Errors: {errors} ({errors/max(total,1)*100:.0f}%)', end='', flush=True)

    print(f'\n\n  ✅ API errors done: {total} requests, {errors} errors ({errors/max(total,1)*100:.0f}% error rate)')
    logger.warning(f'[SCENARIO] API errors complete: {errors}/{total} errors')

    post('/simulate/normal')
    logger.info('[SCENARIO] Returned to normal after API errors')


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3 — MODEL DRIFT
# ─────────────────────────────────────────────────────────────────────────────
def scenario_model_drift(duration=90):
    print_separator('SCENARIO 3: MODEL DRIFT → Performance Degradation')
    logger.critical('[SCENARIO] Starting MODEL DRIFT simulation')

    resp, _ = post('/simulate/model_drift')
    print(f'  Activated: {resp}')
    print(f'\n  Running drift scenario for {duration}s...')
    print('  Watch Grafana → Model Health, Drift Detection, Retraining Triggers\n')

    start = time.time()
    total = 0
    drift_alerts = 0
    accuracy_alerts = 0

    while time.time() - start < duration:
        # Mix of predictions and data ingestion
        for ep, body in [
            ('/predict/regression',    {'fk_type_unite': random.randint(1, 6)}),
            ('/predict/classification', {'fk_type_unite': random.randint(1, 6)}),
            ('/data/ingest',            {}),
        ]:
            resp, status = post(ep, body)
            total += 1

            # Check for drift in data/ingest response
            if ep == '/data/ingest' and isinstance(resp, dict):
                drift_results = resp.get('drift_results', {})
                for feat, info in drift_results.items():
                    if info.get('drift_detected'):
                        drift_alerts += 1
                        logger.warning(f'[DRIFT] Feature {feat} drift detected! score={info.get("score", 0):.3f}')

        time.sleep(random.uniform(0.3, 0.8))
        remaining = int(duration - (time.time() - start))
        print(f'\r  [{remaining:3d}s left] Requests: {total} | Drift alerts: {drift_alerts}', end='', flush=True)

    # Check final monitoring status
    status_resp, _ = get('/monitoring/status')
    if isinstance(status_resp, dict):
        alerts = status_resp.get('alerts', [])
        print(f'\n\n  Active alerts: {len(alerts)}')
        for alert in alerts:
            severity = alert.get('severity', 'unknown').upper()
            print(f'    [{severity}] {alert.get("type")}: {alert.get("message")}')
            logger.critical(f'[ALERT] {severity} — {alert.get("type")}: {alert.get("message")}')

    print(f'\n  ✅ Drift scenario done: {total} requests, {drift_alerts} drift alerts')
    logger.critical(f'[SCENARIO] Model drift complete: {drift_alerts} drift alerts detected')

    post('/simulate/normal')
    logger.info('[SCENARIO] Returned to normal after model drift')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def check_api():
    resp, status = get('/health')
    if status == 200:
        print(f'  ✅ Monitoring API is up: {resp.get("service")}')
        return True
    else:
        print(f'  ❌ Monitoring API not reachable at {BASE_URL}')
        print('     Start it with: python monitoring_api.py')
        return False


def main():
    parser = argparse.ArgumentParser(description='ML Scout Monitoring Simulation Scenarios')
    parser.add_argument('--scenario', choices=['traffic', 'errors', 'drift', 'all'],
                        default='all', help='Scenario to run')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration per scenario in seconds (default: 60)')
    args = parser.parse_args()

    print_separator('ML Scout — Monitoring Simulation (S13)')
    print(f'  Timestamp: {datetime.now().isoformat()}')
    print(f'  Target:    {BASE_URL}')
    print(f'  Scenario:  {args.scenario}')
    print(f'  Duration:  {args.duration}s per scenario')

    if not check_api():
        return

    import os
    os.makedirs('logs', exist_ok=True)

    if args.scenario in ('traffic', 'all'):
        scenario_high_traffic(duration=args.duration)
        time.sleep(5)

    if args.scenario in ('errors', 'all'):
        scenario_api_errors(duration=args.duration)
        time.sleep(5)

    if args.scenario in ('drift', 'all'):
        scenario_model_drift(duration=args.duration)

    print_separator('Simulation Complete')
    print('  Check Grafana at http://localhost:3000 (admin / mlscout2024)')
    print('  Check Prometheus at http://localhost:9090')
    print('  Check Alertmanager at http://localhost:9093')
    print('  Check logs in logs/simulation.log and logs/monitoring.log')
    logger.info('[SIMULATION] All scenarios completed')


if __name__ == '__main__':
    main()
