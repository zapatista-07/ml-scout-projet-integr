# =============================================================================
# 07_mlops_mlflow.py — MLOps avec MLflow
# Tracking des expériences + Model Registry
# Objectif : enregistrer toutes les métriques et modèles automatiquement
# =============================================================================
import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             silhouette_score, davies_bouldin_score)
import joblib

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, 'data')
MODELS  = os.path.join(BASE, 'models')

# ── MLflow config ─────────────────────────────────────────────────────────────
MLFLOW_URI        = f"sqlite:///{os.path.join(BASE, 'mlflow.db')}"
EXPERIMENT_NAME   = "ML_Scout_Projet_Integre"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("█"*65)
print("  MLOPS — MLFLOW TRACKING + MODEL REGISTRY")
print("  Experiment :", EXPERIMENT_NAME)
print("  Tracking URI :", MLFLOW_URI)
print("█"*65)

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================
df_reg = pd.read_csv(os.path.join(DATA, 'dataset_regression.csv'))
df_cls = pd.read_csv(os.path.join(DATA, 'dataset_classification.csv'))
df_clu = pd.read_csv(os.path.join(DATA, 'dataset_clustering.csv'))
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)

# Feature engineering (identique aux scripts précédents)
def build_features(df_r, df_c):
    df = df_r.merge(df_c[['fk_type_unite','season_year','leak_rate']],
                    on=['fk_type_unite','season_year'], how='left')
    df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)
    grp = df.groupby('fk_type_unite')
    df['members_lag1']        = df['nmbr_members_previous_season']
    df['leak_rate_lag1']      = grp['leak_rate'].shift(1).fillna(0)
    df['growth_rate_lag1']    = grp['nbmr_members_season'].pct_change().shift(1).fillna(0)*100
    df['delta_members_lag1']  = grp['nbmr_members_season'].diff().shift(1).fillna(0)
    df['avg_members_unite_past'] = grp['nbmr_members_season'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0]))
    df['season_index']        = grp.cumcount() + 1
    df['retention_rate']      = 1 - df['leak_rate_lag1'] / 100
    df['members_x_retention'] = df['members_lag1'] * df['retention_rate']
    return df

df_feat = build_features(df_reg, df_clu)

FEAT = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
        'delta_members_lag1', 'avg_members_unite_past',
        'season_index', 'fk_type_unite',
        'retention_rate', 'members_x_retention']

X_reg = df_feat[FEAT]
y_reg = df_feat['nbmr_members_season']

# Classification features
df_cls_feat = build_features(
    df_reg,
    df_clu
).merge(df_cls[['fk_type_unite','season_year','dropout_risk']],
        on=['fk_type_unite','season_year'], how='left')
X_cls = df_cls_feat[FEAT]
y_cls = df_cls_feat['dropout_risk'].astype(int)

# =============================================================================
# RUN 1 — RÉGRESSION : RIDGE
# =============================================================================
print("\n" + "="*65)
print("RUN 1 — RÉGRESSION : RIDGE")
print("="*65)

with mlflow.start_run(run_name="Ridge_Regression") as run:
    # Tags
    mlflow.set_tags({
        "section": "D - Regression",
        "model_type": "Ridge",
        "validation": "LOO-CV",
        "dataset": "DWH_18obs",
        "author": "ML Scout"
    })

    # Paramètres
    alpha = 0.01
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("n_features", len(FEAT))
    mlflow.log_param("n_samples", len(X_reg))
    mlflow.log_param("validation", "LOO-CV")
    mlflow.log_param("features", str(FEAT))

    # Entraînement LOO-CV
    loo = LeaveOneOut()
    pipe = mlflow.sklearn.load_model(os.path.join(MODELS, 'regressor_ridge.pkl')) \
           if False else __import__('sklearn.pipeline', fromlist=['Pipeline']).Pipeline([
               ('scaler', StandardScaler()),
               ('model', Ridge(alpha=alpha))
           ])

    y_pred = np.zeros(len(y_reg))
    for tr, te in loo.split(X_reg):
        pipe.fit(X_reg.iloc[tr], y_reg.iloc[tr])
        y_pred[te] = pipe.predict(X_reg.iloc[te])

    pipe.fit(X_reg, y_reg)

    # Métriques
    mse  = mean_squared_error(y_reg, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_reg, y_pred)
    r2   = r2_score(y_reg, y_pred)

    mlflow.log_metric("MSE",  round(mse,3))
    mlflow.log_metric("RMSE", round(rmse,3))
    mlflow.log_metric("MAE",  round(mae,3))
    mlflow.log_metric("R2",   round(r2,3))

    print(f"  Ridge → MSE={mse:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R²={r2:.3f}")

    # Log du modèle
    signature = infer_signature(X_reg, pipe.predict(X_reg))
    mlflow.sklearn.log_model(pipe, "ridge_model",
                             signature=signature,
                             registered_model_name="ML_Scout_Ridge_Regression")

    ridge_run_id = run.info.run_id
    print(f"  Run ID : {ridge_run_id}")

# =============================================================================
# RUN 2 — RÉGRESSION : RANDOM FOREST
# =============================================================================
print("\n" + "="*65)
print("RUN 2 — RÉGRESSION : RANDOM FOREST")
print("="*65)

with mlflow.start_run(run_name="RandomForest_Regression") as run:
    mlflow.set_tags({
        "section": "D - Regression",
        "model_type": "RandomForest",
        "validation": "LOO-CV",
        "dataset": "DWH_18obs"
    })

    params = {'max_depth': 3, 'max_features': 0.9,
              'min_samples_split': 2, 'n_estimators': 100}
    mlflow.log_params(params)
    mlflow.log_param("n_features", len(FEAT))
    mlflow.log_param("n_samples", len(X_reg))

    from sklearn.pipeline import Pipeline
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**params, random_state=42))
    ])

    y_pred_rf = np.zeros(len(y_reg))
    for tr, te in loo.split(X_reg):
        rf_pipe.fit(X_reg.iloc[tr], y_reg.iloc[tr])
        y_pred_rf[te] = rf_pipe.predict(X_reg.iloc[te])

    rf_pipe.fit(X_reg, y_reg)

    mse  = mean_squared_error(y_reg, y_pred_rf)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_reg, y_pred_rf)
    r2   = r2_score(y_reg, y_pred_rf)

    mlflow.log_metric("MSE",  round(mse,3))
    mlflow.log_metric("RMSE", round(rmse,3))
    mlflow.log_metric("MAE",  round(mae,3))
    mlflow.log_metric("R2",   round(r2,3))

    # Feature importance
    fi = rf_pipe.named_steps['model'].feature_importances_
    for fname, fval in zip(FEAT, fi):
        mlflow.log_metric(f"fi_{fname}", round(float(fval),4))

    print(f"  RF → MSE={mse:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R²={r2:.3f}")

    signature = infer_signature(X_reg, rf_pipe.predict(X_reg))
    mlflow.sklearn.log_model(rf_pipe, "rf_model",
                             signature=signature,
                             registered_model_name="ML_Scout_RF_Regression")

# =============================================================================
# RUN 3 — CLASSIFICATION : LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*65)
print("RUN 3 — CLASSIFICATION : LOGISTIC REGRESSION")
print("="*65)

with mlflow.start_run(run_name="LogisticRegression_Classification") as run:
    mlflow.set_tags({
        "section": "C - Classification",
        "model_type": "LogisticRegression",
        "validation": "LOO-CV",
        "target": "dropout_risk"
    })

    from sklearn.pipeline import Pipeline
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(C=0.001, solver='lbfgs',
                                     class_weight='balanced',
                                     random_state=42, max_iter=1000))
    ])

    mlflow.log_param("C", 0.001)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_samples", len(X_cls))

    y_pred_cls = np.zeros(len(y_cls), dtype=int)
    y_prob_cls = np.zeros(len(y_cls))
    for tr, te in loo.split(X_cls):
        lr_pipe.fit(X_cls.iloc[tr], y_cls.iloc[tr])
        y_pred_cls[te] = lr_pipe.predict(X_cls.iloc[te])
        y_prob_cls[te] = lr_pipe.predict_proba(X_cls.iloc[te])[:,1]

    lr_pipe.fit(X_cls, y_cls)

    acc  = accuracy_score(y_cls, y_pred_cls)
    prec = precision_score(y_cls, y_pred_cls, zero_division=0)
    rec  = recall_score(y_cls, y_pred_cls, zero_division=0)
    f1   = f1_score(y_cls, y_pred_cls, zero_division=0)
    auc  = roc_auc_score(y_cls, y_prob_cls)

    mlflow.log_metric("Accuracy",  round(acc,3))
    mlflow.log_metric("Precision", round(prec,3))
    mlflow.log_metric("Recall",    round(rec,3))
    mlflow.log_metric("F1_Score",  round(f1,3))
    mlflow.log_metric("ROC_AUC",   round(auc,3))

    print(f"  LR → Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    signature = infer_signature(X_cls, lr_pipe.predict(X_cls))
    mlflow.sklearn.log_model(lr_pipe, "lr_model",
                             signature=signature,
                             registered_model_name="ML_Scout_LR_Classification")

# =============================================================================
# RUN 4 — CLASSIFICATION : RANDOM FOREST
# =============================================================================
print("\n" + "="*65)
print("RUN 4 — CLASSIFICATION : RANDOM FOREST")
print("="*65)

with mlflow.start_run(run_name="RandomForest_Classification") as run:
    mlflow.set_tags({
        "section": "C - Classification",
        "model_type": "RandomForest",
        "validation": "LOO-CV",
        "target": "dropout_risk"
    })

    from sklearn.pipeline import Pipeline
    rf_cls_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(max_depth=2, min_samples_split=2,
                                         n_estimators=50, class_weight='balanced',
                                         random_state=42))
    ])

    mlflow.log_param("max_depth", 2)
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("class_weight", "balanced")

    y_pred_rf_cls = np.zeros(len(y_cls), dtype=int)
    y_prob_rf_cls = np.zeros(len(y_cls))
    for tr, te in loo.split(X_cls):
        rf_cls_pipe.fit(X_cls.iloc[tr], y_cls.iloc[tr])
        y_pred_rf_cls[te] = rf_cls_pipe.predict(X_cls.iloc[te])
        y_prob_rf_cls[te] = rf_cls_pipe.predict_proba(X_cls.iloc[te])[:,1]

    rf_cls_pipe.fit(X_cls, y_cls)

    acc  = accuracy_score(y_cls, y_pred_rf_cls)
    f1   = f1_score(y_cls, y_pred_rf_cls, zero_division=0)
    auc  = roc_auc_score(y_cls, y_prob_rf_cls)

    mlflow.log_metric("Accuracy", round(acc,3))
    mlflow.log_metric("F1_Score", round(f1,3))
    mlflow.log_metric("ROC_AUC",  round(auc,3))

    print(f"  RF → Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

    signature = infer_signature(X_cls, rf_cls_pipe.predict(X_cls))
    mlflow.sklearn.log_model(rf_cls_pipe, "rf_cls_model",
                             signature=signature,
                             registered_model_name="ML_Scout_RF_Classification")

# =============================================================================
# RUN 5 — CLUSTERING : K-MEANS + GMM
# =============================================================================
print("\n" + "="*65)
print("RUN 5 — CLUSTERING : K-MEANS + GMM")
print("="*65)

# Agrégation par unité
grp_clu = df_clu.groupby('fk_type_unite')
df_unit = grp_clu.agg(
    avg_members   = ('nbmr_members_season', 'mean'),
    std_members   = ('nbmr_members_season', 'std'),
    avg_leak_rate = ('leak_rate',           'mean'),
).reset_index()
df_unit['cv_members']    = df_unit['std_members'] / (df_unit['avg_members'] + 1e-9)
df_unit['avg_variation'] = grp_clu['nbmr_members_season'].apply(lambda x: x.diff().mean()).values

FEAT_CLU = ['avg_variation', 'cv_members']
scaler_clu = StandardScaler()
X_clu = scaler_clu.fit_transform(df_unit[FEAT_CLU])

for model_name, model in [
    ("KMeans_k3", KMeans(n_clusters=3, random_state=42, n_init=10)),
    ("GMM_full_k3", GaussianMixture(n_components=3, covariance_type='full', random_state=42))
]:
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({
            "section": "E - Clustering",
            "model_type": model_name,
            "features": str(FEAT_CLU),
            "n_units": "6"
        })

        labels = model.fit_predict(X_clu)
        sil = silhouette_score(X_clu, labels)
        db  = davies_bouldin_score(X_clu, labels)

        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("features", str(FEAT_CLU))
        mlflow.log_metric("Silhouette_Score",     round(sil,4))
        mlflow.log_metric("Davies_Bouldin_Index", round(db,4))

        print(f"  {model_name} → Silhouette={sil:.4f} DB={db:.4f}")

        if hasattr(model, 'inertia_'):
            mlflow.log_metric("Inertia", round(float(model.inertia_),3))

        mlflow.sklearn.log_model(model, f"{model_name}_model",
                                 registered_model_name=f"ML_Scout_{model_name}")

# =============================================================================
# RUN 6 — XGBOOST TIME SERIES
# =============================================================================
print("\n" + "="*65)
print("RUN 6 — XGBOOST TIME SERIES")
print("="*65)

with mlflow.start_run(run_name="XGBoost_TimeSeries") as run:
    mlflow.set_tags({
        "section": "F - Time Series",
        "model_type": "GradientBoosting_TS",
        "validation": "LOO-CV",
        "forecast_horizon": "2026/2027"
    })

    from sklearn.pipeline import Pipeline
    gb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                             max_depth=2, random_state=42))
    ])

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 2)
    mlflow.log_param("features", str(FEAT))

    y_pred_ts = np.zeros(len(y_reg))
    for tr, te in loo.split(X_reg):
        gb_pipe.fit(X_reg.iloc[tr], y_reg.iloc[tr])
        y_pred_ts[te] = max(gb_pipe.predict(X_reg.iloc[te])[0], 0)

    gb_pipe.fit(X_reg, y_reg)

    mse  = mean_squared_error(y_reg, y_pred_ts)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_reg, y_pred_ts)
    r2   = r2_score(y_reg, y_pred_ts)
    mape = np.mean(np.abs((y_reg + 1e-9 - y_pred_ts) / (y_reg + 1e-9))) * 100

    mlflow.log_metric("MSE",    round(mse,3))
    mlflow.log_metric("RMSE",   round(rmse,3))
    mlflow.log_metric("MAE",    round(mae,3))
    mlflow.log_metric("R2",     round(r2,3))
    mlflow.log_metric("MAPE",   round(mape,2))

    print(f"  XGBoost TS → RMSE={rmse:.3f} MAE={mae:.3f} R²={r2:.3f} MAPE={mape:.1f}%")

    signature = infer_signature(X_reg, gb_pipe.predict(X_reg))
    mlflow.sklearn.log_model(gb_pipe, "xgboost_ts_model",
                             signature=signature,
                             registered_model_name="ML_Scout_XGBoost_TS")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "█"*65)
print("  RÉSUMÉ MLflow")
print("█"*65)

client = mlflow.tracking.MlflowClient()
exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
runs   = client.search_runs(exp.experiment_id,
                             order_by=["start_time DESC"])

print(f"\n  {len(runs)} runs enregistrés dans '{EXPERIMENT_NAME}'")
print(f"\n  {'Run Name':35s} {'Métriques clés'}")
print("  " + "-"*65)
for r in runs:
    name    = r.data.tags.get('mlflow.runName', r.info.run_id[:8])
    metrics = r.data.metrics
    if 'R2' in metrics:
        key = f"R²={metrics['R2']:.3f}  RMSE={metrics.get('RMSE',0):.3f}"
    elif 'F1_Score' in metrics:
        key = f"F1={metrics['F1_Score']:.3f}  AUC={metrics.get('ROC_AUC',0):.3f}"
    elif 'Silhouette_Score' in metrics:
        key = f"Silhouette={metrics['Silhouette_Score']:.4f}  DB={metrics.get('Davies_Bouldin_Index',0):.4f}"
    else:
        key = str(metrics)
    print(f"  {name:35s} {key}")

print(f"""
  Pour voir l'UI MLflow :
  → Dans ton terminal : mlflow ui --backend-store-uri sqlite:///mlflow.db
  → Ouvre : http://localhost:5000  (ou 5001 si Flask tourne déjà)
  → mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
""")

print("█"*65)
print("  ✅ MLOPS MLFLOW TERMINÉ")
print("█"*65)
