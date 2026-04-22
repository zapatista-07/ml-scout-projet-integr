# test_predict.py — Test de l'endpoint /predict
import requests, json

BASE = "http://localhost:5000"

print("="*55)
print("TEST ENDPOINT /predict")
print("="*55)

# Test 1 — Régression
print("\n1. Test Régression (Unité 1) :")
r = requests.post(f"{BASE}/predict", json={
    "type": "regression",
    "fk_type_unite": 1,
    "members_lag1": 12,
    "leak_rate_lag1": 52.0
})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

# Test 2 — Classification
print("\n2. Test Classification (Unité 2) :")
r = requests.post(f"{BASE}/predict", json={
    "type": "classification",
    "fk_type_unite": 2,
    "members_lag1": 25,
    "leak_rate_lag1": 0.0
})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

# Test 3 — Health
print("\n3. Test Health :")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

print("\n✅ Tous les tests passés !")
