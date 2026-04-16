# =============================================================================
# 02_regression.py — Section D : Régression (≥ 2 modèles)
# Objectif : Prédire nbmr_members_season + forecast 2026/2027
# Dataset  : 18 observations DWH avec features laggées enrichies
# =============================================================================
import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs('models',  exist_ok=True)
os.makedirs('visuals', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# D.0  CHARGEMENT & FEATURE ENGINEERING COMPLET (18 lignes DWH)
# ─────────────────────────────────────────────────────────────────────────────
df_reg = pd.read_csv('data/dataset_regression.csv')
df_clu = pd.read_csv('data/dataset_clustering.csv')

# Fusionner pour avoir leak_rate
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)
df = df_reg.merge(df_clu[['fk_type_unite','season_year','leak_rate']],
                  on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

print("="*65)
print("D.0  DONNÉES & FEATURE ENGINEERING (DWH — 18 lignes)")
print("="*65)
print(df.to_string(index=False))

# ── Feature Engineering ───────────────────────────────────────────────────────
grp = df.groupby('fk_type_unite')

# 1. members_lag1 : membres saison précédente (= nmbr_members_previous_season)
df['members_lag1'] = df['nmbr_members_previous_season']

# 2. leak_rate_lag1 : taux de fuite saison précédente
df['leak_rate_lag1'] = grp['leak_rate'].shift(1).fillna(0)

# 3. growth_rate_lag1 : taux de croissance saison précédente
df['growth_rate_lag1'] = grp['nbmr_members_season'].pct_change().shift(1).fillna(0) * 100

# 4. delta_members_lag1 : variation absolue saison précédente
df['delta_members_lag1'] = grp['nbmr_members_season'].diff().shift(1).fillna(0)

# 5. avg_members_unite_past : moyenne historique expanding (sans fuite du futur)
df['avg_members_unite_past'] = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0])
)

# 6. season_index : rang de la saison (1, 2, 3)
df['season_index'] = grp.cumcount() + 1

# 7. fk_type_unite : déjà présent

# 8. retention_rate : taux de rétention (1 - leak_rate/100)
df['retention_rate'] = 1 - df['leak_rate_lag1'] / 100

# 9. members_x_retention : interaction forte (membres × rétention)
df['members_x_retention'] = df['members_lag1'] * df['retention_rate']

FEATURE_COLS = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
                'delta_members_lag1', 'avg_members_unite_past',
                'season_index', 'fk_type_unite',
                'retention_rate', 'members_x_retention']
TARGET = 'nbmr_members_season'

X = df[FEATURE_COLS]
y = df[TARGET]

print(f"\n  Features ({len(FEATURE_COLS)}) : {FEATURE_COLS}")
print(f"\n  Dataset enrichi :")
show = ['fk_type_unite','season_year'] + FEATURE_COLS + [TARGET]
print(df[show].round(2).to_string(index=False))

# Corrélations
print(f"\n  Corrélations avec {TARGET} :")
corr = df[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(corr.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# D.1  EXPLICATION DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("D.1  EXPLICATION DES MODÈLES")
print("="*65)
print("""
Approche : Features Laggées + Feature Engineering
  "Pas de nouvelles données — uniquement des indicateurs dérivés du DWH."
  members_lag1        → membres saison précédente (corrélation forte)
  leak_rate_lag1      → taux de fuite saison précédente
  growth_rate_lag1    → taux de croissance saison précédente
  delta_members_lag1  → variation absolue saison précédente
  avg_members_unite_past → moyenne historique de l'unité
  season_index        → numéro de saison (tendance temporelle)
  fk_type_unite       → identité de l'unité
  retention_rate      → 1 - leak_rate/100 (taux de fidélisation)
  members_x_retention → interaction membres × rétention (feature clé)

Modèle 1 — Ridge Regression (régression linéaire régularisée L2)
  Intuition  : minimise les résidus² + pénalité L2 sur les coefficients.
               Réduit le surapprentissage en contraignant les poids.
  Paramètres : alpha (force de régularisation) — sélectionné par GridSearch.
  Hypothèses : linéarité, homoscédasticité, résidus normaux, indépendance.
  Limites    : ne capture pas les non-linéarités.
  Justification : baseline interprétable, robuste à la multicolinéarité.

Modèle 2 — Random Forest Regressor (ensemble d'arbres, bagging)
  Intuition  : agrège N arbres entraînés sur sous-échantillons aléatoires.
               La moyenne des prédictions réduit la variance.
  Paramètres : n_estimators, max_depth, min_samples_split — GridSearch LOO.
  Hypothèses : aucune hypothèse distributionnelle.
  Limites    : risque de surapprentissage sur petits datasets.
  Justification : capture les non-linéarités et interactions entre features.

  ⚠️  Validation : Leave-One-Out CV (18 obs. → 17 train / 1 test × 18 iter.)
""")

# ─────────────────────────────────────────────────────────────────────────────
# D.2  PIPELINES + GRIDSEARCH (LOO-CV)
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("D.2  PIPELINES + GRIDSEARCH (LOO-CV)")
print("="*65)

loo = LeaveOneOut()

# Ridge
ridge_pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
ridge_gs = GridSearchCV(ridge_pipe,
                        {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 50, 100]},
                        cv=loo, scoring='neg_root_mean_squared_error', n_jobs=-1)
ridge_gs.fit(X, y)
best_ridge = ridge_gs.best_estimator_
print(f"\n  Ridge — meilleur alpha : {ridge_gs.best_params_['model__alpha']}"
      f"  (RMSE LOO : {-ridge_gs.best_score_:.3f})")

# Random Forest
rf_pipe = Pipeline([('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {
    'model__n_estimators':     [100, 200],
    'model__max_depth':        [2, 3, 4],
    'model__min_samples_split':[2, 3],
    'model__max_features':     ['sqrt', 0.7, 0.9],
}, cv=loo, scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_gs.fit(X, y)
best_rf = rf_gs.best_estimator_
print(f"  RF    — meilleurs params : {rf_gs.best_params_}"
      f"\n          RMSE LOO : {-rf_gs.best_score_:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# D.3  VALIDATION LOO-CV — MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("D.3  VALIDATION LOO-CV — MÉTRIQUES")
print("="*65)
print("  Justification : 18 obs. → LOO = 17 train / 1 test × 18 itérations.\n")

results_rows = []
all_loo = {}

for name, pipe in [('Ridge', best_ridge), ('Random Forest', best_rf)]:
    y_pred_loo = np.zeros(len(y))
    for tr, te in loo.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred_loo[te] = pipe.predict(X.iloc[te])

    y_pred_loo = np.maximum(y_pred_loo, 0)
    mse  = mean_squared_error(y, y_pred_loo)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y, y_pred_loo)
    r2   = r2_score(y, y_pred_loo)
    all_loo[name] = y_pred_loo

    results_rows.append({'Modèle': name, 'MSE': round(mse,3),
                         'RMSE': round(rmse,3), 'MAE': round(mae,3),
                         'R²': round(r2,3)})
    print(f"  {name} :")
    print(f"    MSE={mse:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

results_df = pd.DataFrame(results_rows)
results_df.to_csv('models/regression_comparison.csv', index=False)
best_name = results_df.loc[results_df['R²'].idxmax(), 'Modèle']
print(f"\n  ✅ Meilleur modèle (R² LOO max) : {best_name}")

# ─────────────────────────────────────────────────────────────────────────────
# D.4  PRÉDICTION 2026/2027
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("D.4  PRÉDICTION 2026/2027")
print("="*65)

# Entraîner sur tout le dataset
for name, pipe in [('Ridge', best_ridge), ('Random Forest', best_rf)]:
    pipe.fit(X, y)

# Construire les features pour 2026/2027 à partir des données 2025/2026
pred_rows = []
for unit in sorted(df['fk_type_unite'].unique()):
    hist = df[df['fk_type_unite'] == unit].sort_values('season_year')
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]

    m_curr = last['nbmr_members_season']
    m_prev = prev['nbmr_members_season']
    growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta  = m_curr - m_prev
    leak1  = last['leak_rate']
    ret    = 1 - leak1 / 100

    pred_rows.append({
        'fk_type_unite':          unit,
        'season':                 '2026/2027',
        'members_lag1':           m_curr,
        'leak_rate_lag1':         leak1,
        'growth_rate_lag1':       growth,
        'delta_members_lag1':     delta,
        'avg_members_unite_past': hist['nbmr_members_season'].mean(),
        'season_index':           4,
        'retention_rate':         ret,
        'members_x_retention':    m_curr * ret,
    })

df_future = pd.DataFrame(pred_rows)
X_future  = df_future[FEATURE_COLS]

df_future['pred_ridge'] = np.maximum(best_ridge.predict(X_future).round(1), 0)
df_future['pred_rf']    = np.maximum(best_rf.predict(X_future).round(1), 0)

print(f"\n  {'Unité':>6} | {'Réel 2025/26':>12} | {'Ridge':>8} | {'RF':>8} | {'Tendance RF'}")
print("  " + "-"*58)
for _, row in df_future.iterrows():
    real = df[df['fk_type_unite']==row['fk_type_unite']].sort_values('season_year')['nbmr_members_season'].iloc[-1]
    delta_rf = row['pred_rf'] - real
    trend = "↗ croissance" if delta_rf > 1 else ("↘ déclin" if delta_rf < -1 else "→ stable")
    print(f"  {int(row['fk_type_unite']):>6} | {real:>12.0f} | "
          f"{row['pred_ridge']:>8.1f} | {row['pred_rf']:>8.1f} | {trend}")

df_future.to_csv('models/predictions_2026_2027.csv', index=False)
print(f"\n  💾 models/predictions_2026_2027.csv")

# ─────────────────────────────────────────────────────────────────────────────
# D.5  VÉRIFICATION HYPOTHÈSES RIDGE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("D.5  VÉRIFICATION HYPOTHÈSES — RIDGE")
print("="*65)
ridge_resid = y.values - all_loo['Ridge']
stat_sw, p_sw = stats.shapiro(ridge_resid)
corr_r, p_corr = stats.pearsonr(all_loo['Ridge'], ridge_resid)
print(f"  Shapiro-Wilk (normalité résidus) : W={stat_sw:.4f}, p={p_sw:.4f}"
      f"  → {'✅ normaux' if p_sw>0.05 else '⚠️ non-normaux (petit échantillon)'}")
print(f"  Corrélation prédits/résidus      : r={corr_r:.4f}, p={p_corr:.4f}"
      f"  → {'✅ homoscédastique' if p_corr>0.05 else '⚠️ hétéroscédastique'}")

# ─────────────────────────────────────────────────────────────────────────────
# D.6  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Résidus + Q-Q
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for idx, (name, yp) in enumerate(all_loo.items()):
    resid = y.values - yp
    color = 'steelblue' if idx == 0 else 'coral'
    axes[0][idx].scatter(yp, resid, color=color, edgecolors='k', s=80, alpha=0.85)
    axes[0][idx].axhline(0, color='red', linestyle='--', linewidth=1.5)
    for i, (xi, yi_r) in enumerate(zip(yp, resid)):
        axes[0][idx].annotate(f"U{int(df['fk_type_unite'].iloc[i])}",
                              (xi, yi_r), textcoords='offset points',
                              xytext=(4,3), fontsize=8, alpha=0.8)
    axes[0][idx].set_xlabel('Valeurs Prédites (LOO)')
    axes[0][idx].set_ylabel('Résidus')
    axes[0][idx].set_title(f'Résidus vs Prédits — {name}\nR²={r2_score(y,yp):.3f}')
    axes[0][idx].grid(alpha=0.3)
    stats.probplot(resid, dist='norm', plot=axes[1][idx])
    axes[1][idx].set_title(f'Q-Q Plot — {name}')
    axes[1][idx].grid(alpha=0.3)
plt.suptitle('Analyse des Résidus — Régression (LOO-CV, 18 obs. DWH)', fontsize=13)
plt.tight_layout()
plt.savefig('visuals/residual_plots_regression.png', dpi=150)
plt.close()
print("\n  💾 visuals/residual_plots_regression.png")

# Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for idx, (name, yp) in enumerate(all_loo.items()):
    color = 'steelblue' if idx == 0 else 'coral'
    axes[idx].scatter(y, yp, color=color, edgecolors='k', s=90, alpha=0.85, zorder=3)
    lims = [min(y.min(), yp.min())-1, max(y.max(), yp.max())+1]
    axes[idx].plot(lims, lims, 'r--', linewidth=1.5, label='Parfait')
    for i, (xi, yi_p) in enumerate(zip(y, yp)):
        axes[idx].annotate(f"U{int(df['fk_type_unite'].iloc[i])}",
                           (xi, yi_p), textcoords='offset points',
                           xytext=(4,3), fontsize=8, alpha=0.8)
    axes[idx].set_xlabel('Réel'); axes[idx].set_ylabel('Prédit (LOO)')
    axes[idx].set_title(f'Actual vs Predicted — {name}\nR²={r2_score(y,yp):.3f}')
    axes[idx].legend(); axes[idx].grid(alpha=0.3)
plt.suptitle('Actual vs Predicted — Régression (LOO-CV, 18 obs. DWH)', fontsize=13)
plt.tight_layout()
plt.savefig('visuals/actual_vs_predicted_regression.png', dpi=150)
plt.close()
print("  💾 visuals/actual_vs_predicted_regression.png")

# Feature importance RF
best_rf.fit(X, y)
rf_obj = best_rf.named_steps['model']
fi = pd.Series(rf_obj.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
colors_fi = ['#1565C0' if v >= fi.mean() else '#90CAF9' for v in fi.sort_values()]
fi.sort_values().plot(kind='barh', ax=ax, color=colors_fi)
ax.axvline(fi.mean(), color='red', linestyle='--', label=f'Moyenne ({fi.mean():.3f})')
ax.set_title('Feature Importance — Random Forest Regressor\n(Features Laggées DWH)')
ax.set_xlabel('Importance (Gini)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/feature_importance_regression.png', dpi=150)
plt.close()
print("  💾 visuals/feature_importance_regression.png")
print(f"\n  Feature Importance RF :\n{fi.round(4).to_string()}")

# Coefficients Ridge
best_ridge.fit(X, y)
ridge_obj = best_ridge.named_steps['model']
coefs = pd.Series(ridge_obj.coef_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
coefs.plot(kind='barh', ax=ax,
           color=['coral' if v < 0 else 'steelblue' for v in coefs])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Coefficients — Ridge Regression (standardisés)\n(Features Laggées DWH)')
ax.set_xlabel('Coefficient')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/coefficients_ridge.png', dpi=150)
plt.close()
print("  💾 visuals/coefficients_ridge.png")

# Forecast 2026/2027
fig, ax = plt.subplots(figsize=(11, 5))
units = sorted(df['fk_type_unite'].unique())
x = np.arange(len(units))
w = 0.25
hist_vals  = [df[df['fk_type_unite']==u].sort_values('season_year')['nbmr_members_season'].iloc[-1] for u in units]
ridge_vals = df_future.sort_values('fk_type_unite')['pred_ridge'].values
rf_vals    = df_future.sort_values('fk_type_unite')['pred_rf'].values
ax.bar(x-w, hist_vals,  w, label='Réel 2025/2026',  color='#90CAF9', alpha=0.9)
ax.bar(x,   ridge_vals, w, label='Ridge 2026/2027',  color='steelblue', alpha=0.9)
ax.bar(x+w, rf_vals,    w, label='RF 2026/2027',     color='coral', alpha=0.9)
for i, (h, r, f) in enumerate(zip(hist_vals, ridge_vals, rf_vals)):
    ax.text(i-w, h+0.3, str(int(h)), ha='center', fontsize=8)
    ax.text(i,   r+0.3, f'{r:.0f}', ha='center', fontsize=8)
    ax.text(i+w, f+0.3, f'{f:.0f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'Unité {u}' for u in units])
ax.set_ylabel("Nombre d'adhérents")
ax.set_title('Prédiction 2026/2027 par Unité\n(Features Laggées DWH)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/forecast_2026_regression.png', dpi=150)
plt.close()
print("  💾 visuals/forecast_2026_regression.png")

# Comparaison métriques
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for i, metric in enumerate(['MSE', 'RMSE', 'MAE', 'R²']):
    vals = results_df[metric].values
    bars = axes[i].bar(results_df['Modèle'], vals,
                       color=['steelblue', 'coral'], width=0.5, alpha=0.85)
    axes[i].set_title(metric)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, v*1.03,
                     f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(axis='y', alpha=0.3)
plt.suptitle('Comparaison Ridge vs Random Forest — Métriques LOO-CV\n'
             '(Features Laggées, 18 obs. DWH)', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/regression_metrics_comparison.png', dpi=150)
plt.close()
print("  💾 visuals/regression_metrics_comparison.png")

joblib.dump(best_ridge, 'models/regressor_ridge.pkl')
joblib.dump(best_rf,    'models/regressor_rf.pkl')

print("\n" + "="*65)
print("✅ RÉGRESSION TERMINÉE")
print("="*65)
