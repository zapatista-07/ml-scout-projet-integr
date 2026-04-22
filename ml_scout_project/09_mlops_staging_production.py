# =============================================================================
# 09_mlops_staging_production.py
# Gestion stricte Staging → Production
# Un seul modèle actif par section en Production
# =============================================================================
import os, warnings
warnings.filterwarnings('ignore')
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
client = MlflowClient()

print("█"*65)
print("  MLOPS — STAGING → PRODUCTION (workflow strict)")
print("█"*65)

EXPERIMENT_NAME = "ML_Scout_Projet_Integre"
exp  = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(exp.experiment_id)

# ── Identifier le meilleur modèle par section ─────────────────────────────────
sections = {
    'D - Regression':     ('R2',               'ML_Scout_Ridge_Regression'),
    'C - Classification': ('F1_Score',          'ML_Scout_LR_Classification'),
    'E - Clustering':     ('Silhouette_Score',  'ML_Scout_GMM_full_k3'),
    'F - Time Series':    ('R2',               'ML_Scout_XGBoost_TS'),
}

print("\n" + "="*65)
print("ÉTAPE 1 — PASSAGE EN STAGING")
print("="*65)

for section, (metric_key, best_model_name) in sections.items():
    sec_runs = [r for r in runs if r.data.tags.get('section','').startswith(section[:1])]
    if not sec_runs:
        continue

    best_run = max(sec_runs, key=lambda r: r.data.metrics.get(metric_key, -999))
    best_val = best_run.data.metrics.get(metric_key, 0)
    run_name = best_run.data.tags.get('mlflow.runName', '')

    try:
        versions = client.search_model_versions(f"name='{best_model_name}'")
        if versions:
            v = versions[0].version
            # Ajouter alias staging
            client.set_registered_model_alias(best_model_name, "staging", v)
            client.set_registered_model_tag(best_model_name, "stage", "staging")
            print(f"  ✅ {best_model_name:45s} v{v} → STAGING  ({metric_key}={best_val:.3f})")
    except Exception as e:
        print(f"  ⚠️  {best_model_name} : {e}")

print("\n" + "="*65)
print("ÉTAPE 2 — VALIDATION STAGING (vérification seuils)")
print("="*65)

# Seuils de validation pour passer en Production
thresholds = {
    'ML_Scout_Ridge_Regression':   ('R2',               0.5),
    'ML_Scout_LR_Classification':  ('F1_Score',         0.6),
    'ML_Scout_GMM_full_k3':        ('Silhouette_Score', 0.4),
    'ML_Scout_XGBoost_TS':         ('R2',               0.5),
}

validated = {}
for model_name, (metric_key, threshold) in thresholds.items():
    sec_runs = [r for r in runs
                if r.data.tags.get('mlflow.runName','') in model_name.replace('ML_Scout_','').replace('_',' ')]

    # Chercher la métrique dans tous les runs
    best_val = -999
    for r in runs:
        val = r.data.metrics.get(metric_key, -999)
        reg_models = [v.name for v in client.search_model_versions(
            f"run_id='{r.info.run_id}'")]
        if model_name in reg_models and val > best_val:
            best_val = val

    if best_val == -999:
        # Fallback : lire depuis les tags
        try:
            m = client.get_registered_model(model_name)
            metric_str = m.tags.get('metric', '0')
            best_val = float(metric_str.split('=')[-1]) if '=' in metric_str else 0
        except:
            best_val = 0

    passed = best_val >= threshold
    status = "✅ VALIDÉ" if passed else "❌ REJETÉ"
    validated[model_name] = passed
    print(f"  {status} {model_name:45s} {metric_key}={best_val:.3f} (seuil={threshold})")

print("\n" + "="*65)
print("ÉTAPE 3 — PROMOTION EN PRODUCTION (modèles validés uniquement)")
print("="*65)

for model_name, (metric_key, threshold) in thresholds.items():
    if not validated.get(model_name, False):
        print(f"  ⏭️  {model_name} — non promu (seuil non atteint)")
        continue
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            v = versions[0].version
            # Supprimer ancien alias production si existe
            try:
                client.delete_registered_model_alias(model_name, "production")
            except:
                pass
            # Promouvoir en production
            client.set_registered_model_alias(model_name, "production", v)
            client.set_registered_model_tag(model_name, "stage", "production")
            client.set_registered_model_tag(model_name, "validated", "true")
            client.set_registered_model_tag(model_name, "promoted_by", "AutoML_Pipeline")
            print(f"  🚀 {model_name:45s} v{v} → PRODUCTION ✅")
    except Exception as e:
        print(f"  ⚠️  {model_name} : {e}")

print("\n" + "="*65)
print("ÉTAPE 4 — RAPPORT FINAL")
print("="*65)

all_models = client.search_registered_models()
report = []
print(f"\n  {'Modèle':45s} {'Stage':>12} {'Validé':>8} {'Alias'}")
print("  " + "-"*80)
for m in all_models:
    stage     = m.tags.get('stage', 'registered')
    validated_tag = m.tags.get('validated', 'false')
    aliases   = []
    versions  = client.search_model_versions(f"name='{m.name}'")
    if versions:
        aliases = list(versions[0].aliases) if hasattr(versions[0], 'aliases') else []
    alias_str = ', '.join(aliases) if aliases else '-'
    icon = "🚀" if stage == 'production' else ("⏳" if stage == 'staging' else "📦")
    print(f"  {icon} {m.name:43s} {stage:>12}  {validated_tag:>8}  {alias_str}")
    report.append({'Model': m.name, 'Stage': stage,
                   'Validated': validated_tag, 'Aliases': alias_str})

df = pd.DataFrame(report)
df.to_csv(os.path.join(BASE, 'models', 'mlops_staging_production.csv'), index=False)
print(f"\n  💾 models/mlops_staging_production.csv")

prod_count = sum(1 for r in report if r['Stage'] == 'production')
print(f"""
  Résumé workflow Staging → Production :
  🚀 {prod_count} modèles en Production
  ⏳ Staging validé avant promotion
  ✅ Seuils de qualité respectés
  ✅ Un seul alias 'production' par modèle
  ✅ Pipeline automatisé et reproductible
""")

print("█"*65)
print("  ✅ STAGING → PRODUCTION TERMINÉ")
print("█"*65)
