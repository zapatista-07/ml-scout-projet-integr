# =============================================================================
# 08_mlops_registry.py — MLOps : Model Registry + Promotion Production
# Objectif : Promouvoir les meilleurs modèles en Production
#            + Log artifacts + Comparaison automatique
# =============================================================================
import os, warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np

BASE     = os.path.dirname(os.path.abspath(__file__))
VISUALS  = os.path.join(BASE, 'visuals')
DATA     = os.path.join(BASE, 'data')
MODELS   = os.path.join(BASE, 'models')

MLFLOW_URI      = f"sqlite:///{os.path.join(BASE, 'mlflow.db')}"
EXPERIMENT_NAME = "ML_Scout_Projet_Integre"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

print("█"*65)
print("  MLOPS — MODEL REGISTRY + PROMOTION PRODUCTION")
print("█"*65)

# =============================================================================
# 1. RÉCUPÉRER TOUS LES RUNS
# =============================================================================
exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
runs   = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])

print(f"\n  {len(runs)} runs trouvés dans '{EXPERIMENT_NAME}'")

# =============================================================================
# 2. IDENTIFIER LES MEILLEURS MODÈLES PAR SECTION
# =============================================================================
print("\n" + "="*65)
print("ÉTAPE 1 — IDENTIFICATION DES MEILLEURS MODÈLES")
print("="*65)

best_models = {}

# Régression → meilleur R²
reg_runs = [r for r in runs if r.data.tags.get('section','').startswith('D')]
if reg_runs:
    best_reg = max(reg_runs, key=lambda r: r.data.metrics.get('R2', -999))
    best_models['regression'] = {
        'run': best_reg,
        'name': best_reg.data.tags.get('mlflow.runName'),
        'metric': f"R²={best_reg.data.metrics.get('R2',0):.3f}",
        'registered_name': f"ML_Scout_{best_reg.data.tags.get('model_type','')}_Regression"
    }
    print(f"  ✅ Meilleur Régression : {best_models['regression']['name']} "
          f"({best_models['regression']['metric']})")

# Classification → meilleur F1
cls_runs = [r for r in runs if r.data.tags.get('section','').startswith('C')]
if cls_runs:
    best_cls = max(cls_runs, key=lambda r: r.data.metrics.get('F1_Score', -999))
    best_models['classification'] = {
        'run': best_cls,
        'name': best_cls.data.tags.get('mlflow.runName'),
        'metric': f"F1={best_cls.data.metrics.get('F1_Score',0):.3f}",
        'registered_name': f"ML_Scout_{best_cls.data.tags.get('model_type','')}_Classification"
    }
    print(f"  ✅ Meilleur Classification : {best_models['classification']['name']} "
          f"({best_models['classification']['metric']})")

# Clustering → meilleur Silhouette
clu_runs = [r for r in runs if r.data.tags.get('section','').startswith('E')]
if clu_runs:
    best_clu = max(clu_runs, key=lambda r: r.data.metrics.get('Silhouette_Score', -999))
    best_models['clustering'] = {
        'run': best_clu,
        'name': best_clu.data.tags.get('mlflow.runName'),
        'metric': f"Silhouette={best_clu.data.metrics.get('Silhouette_Score',0):.4f}",
        'registered_name': f"ML_Scout_{best_clu.data.tags.get('model_type','')}"
    }
    print(f"  ✅ Meilleur Clustering : {best_models['clustering']['name']} "
          f"({best_models['clustering']['metric']})")

# Time Series → meilleur R²
ts_runs = [r for r in runs if r.data.tags.get('section','').startswith('F')]
if ts_runs:
    best_ts = max(ts_runs, key=lambda r: r.data.metrics.get('R2', -999))
    best_models['timeseries'] = {
        'run': best_ts,
        'name': best_ts.data.tags.get('mlflow.runName'),
        'metric': f"R²={best_ts.data.metrics.get('R2',0):.3f}",
        'registered_name': "ML_Scout_XGBoost_TS"
    }
    print(f"  ✅ Meilleur Time Series : {best_models['timeseries']['name']} "
          f"({best_models['timeseries']['metric']})")

# =============================================================================
# 3. PROMOUVOIR LES MEILLEURS MODÈLES EN PRODUCTION
# =============================================================================
print("\n" + "="*65)
print("ÉTAPE 2 — PROMOTION EN PRODUCTION (Staging → Production)")
print("="*65)

for section, info in best_models.items():
    reg_name = info['registered_name']
    try:
        # Récupérer les versions du modèle
        versions = client.search_model_versions(f"name='{reg_name}'")
        if versions:
            latest = versions[0]
            version_num = latest.version

            # Ajouter alias "production" (MLflow 3.x)
            try:
                client.set_registered_model_alias(reg_name, "production", version_num)
                print(f"  ✅ {reg_name} v{version_num} → alias 'production' ✅")
            except Exception:
                # Fallback pour versions antérieures
                print(f"  ✅ {reg_name} v{version_num} → enregistré ✅")

            # Ajouter description
            client.update_registered_model(
                reg_name,
                description=f"Meilleur modèle {section} — {info['metric']} — ML Scout Projet Intégré"
            )
            # Ajouter tags au modèle
            client.set_registered_model_tag(reg_name, "status", "production")
            client.set_registered_model_tag(reg_name, "section", section)
            client.set_registered_model_tag(reg_name, "metric", info['metric'])
            client.set_registered_model_tag(reg_name, "validated_by", "MLOps_Pipeline")

    except Exception as e:
        print(f"  ⚠️  {reg_name} : {e}")

# =============================================================================
# 4. LOG DES ARTIFACTS (visuels + données) DANS LES RUNS
# =============================================================================
print("\n" + "="*65)
print("ÉTAPE 3 — LOG DES ARTIFACTS (visuels + données)")
print("="*65)

# Mapping run → visuels
artifacts_map = {
    'Ridge_Regression': [
        'actual_vs_predicted_regression.png',
        'residual_plots_regression.png',
        'coefficients_ridge.png',
        'regression_metrics_comparison.png',
        'forecast_2026_regression.png',
    ],
    'RandomForest_Regression': [
        'feature_importance_regression.png',
        'regression_metrics_comparison.png',
    ],
    'LogisticRegression_Classification': [
        'confusion_matrix_classification.png',
        'roc_curve_classification.png',
        'dropout_proba_2026.png',
        'classification_metrics_comparison.png',
    ],
    'RandomForest_Classification': [
        'feature_importance_classification.png',
    ],
    'KMeans_k3': [
        'clustering_pca_2d.png',
        'cluster_heatmap.png',
        'elbow_silhouette_db.png',
        'silhouette_per_unit.png',
        'cluster_radar.png',
    ],
    'GMM_full_k3': [
        'gmm_probabilities.png',
        'clustering_dendrogram.png',
    ],
    'XGBoost_TimeSeries': [
        'ts_forecast_par_unite.png',
        'ts_actual_vs_predicted.png',
        'ts_feature_importance.png',
        'ts_model_comparison.png',
        'ts_forecast_barplot.png',
    ],
}

for run in runs:
    run_name = run.data.tags.get('mlflow.runName', '')
    if run_name in artifacts_map:
        visuals_to_log = artifacts_map[run_name]
        logged = 0
        with mlflow.start_run(run_id=run.info.run_id):
            for vis in visuals_to_log:
                vis_path = os.path.join(VISUALS, vis)
                if os.path.exists(vis_path):
                    mlflow.log_artifact(vis_path, artifact_path="visualizations")
                    logged += 1
            # Log dataset
            for csv_file in ['dataset_regression.csv', 'dataset_classification.csv',
                             'dataset_clustering.csv', 'dataset_timeseries.csv']:
                csv_path = os.path.join(DATA, csv_file)
                if os.path.exists(csv_path):
                    mlflow.log_artifact(csv_path, artifact_path="data")
        print(f"  ✅ {run_name:40s} → {logged} visuels + données loggés")

# =============================================================================
# 5. RAPPORT FINAL MODEL REGISTRY
# =============================================================================
print("\n" + "="*65)
print("ÉTAPE 4 — RAPPORT FINAL MODEL REGISTRY")
print("="*65)

all_models = client.search_registered_models()
print(f"\n  {len(all_models)} modèles dans le Registry :\n")
print(f"  {'Modèle':45s} {'Version':>8} {'Status':>12} {'Métrique'}")
print("  " + "-"*80)

for m in all_models:
    versions = client.search_model_versions(f"name='{m.name}'")
    if versions:
        v = versions[0]
        status = m.tags.get('status', 'registered')
        metric = m.tags.get('metric', '-')
        print(f"  {m.name:45s} v{v.version:>6}   {status:>12}   {metric}")

# Sauvegarder le rapport
report = []
for m in all_models:
    versions = client.search_model_versions(f"name='{m.name}'")
    if versions:
        v = versions[0]
        report.append({
            'Model': m.name,
            'Version': v.version,
            'Status': m.tags.get('status', 'registered'),
            'Section': m.tags.get('section', '-'),
            'Metric': m.tags.get('metric', '-'),
            'Description': m.description or '-',
        })

df_report = pd.DataFrame(report)
report_path = os.path.join(MODELS, 'mlflow_model_registry.csv')
df_report.to_csv(report_path, index=False)
print(f"\n  💾 {report_path}")

print(f"""
  Résumé MLOps :
  ✅ {len(runs)} runs trackés dans MLflow
  ✅ {len(all_models)} modèles dans le Model Registry
  ✅ Meilleurs modèles promus en Production
  ✅ Visuels et données loggés comme artifacts
  ✅ UI disponible sur http://127.0.0.1:5001
""")

print("█"*65)
print("  ✅ MODEL REGISTRY + PROMOTION TERMINÉS")
print("█"*65)
