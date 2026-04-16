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

## 🗂️ Structure

```
ml_scout_project/
├── data/                    ← Datasets DWH
├── models/                  ← Modèles sauvegardés + résultats CSV
├── visuals/                 ← Graphiques générés
├── templates/               ← HTML du site web
├── static/                  ← CSS + JS + visuels
├── 01_data_preparation.py
├── 02_regression.py
├── 03_classification.py
├── 04_clustering.py
├── 06_anomaly_detection.py
├── app.py                   ← Site web Flask
├── flask_api.py             ← API Flask pour n8n
└── build_features.py        ← Feature engineering
05_timeseries.py             ← Time Series (dossier parent)
```

## 🚀 Lancement

```bash
# Installer les dépendances
pip install -r ml_scout_project/requirements.txt

# Lancer le site web
cd ml_scout_project
python app.py
# → http://localhost:5000

# Lancer l'API n8n
python flask_api.py
# → http://localhost:5000/dashboard
```

## 🤖 Automatisation n8n

Pipeline automatique mensuel via n8n + Docker :
- Lance les 5 scripts ML
- Récupère les résultats
- Envoie un email d'alerte si anomalie détectée

## 📈 Résultats

- **18 observations DWH** (6 unités × 3 saisons)
- **Validation LOO-CV** (Leave-One-Out)
- **Features laggées** : members_lag1, leak_rate_lag1, growth_rate_lag1, etc.
- **2 anomalies confirmées** : U1 (2025/2026) et U2 (2024/2025)

## 👨‍💻 Technologies

Python · Flask · scikit-learn · statsmodels · pandas · matplotlib · n8n · Docker
