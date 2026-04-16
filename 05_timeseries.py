# =============================================================================
# 05_timeseries.py  —  Section F : Time Series / Forecasting
# Objectif : Prédire le nombre futur d'adhérents par unité et par saison
# Dataset  : 18 observations DWH (6 unités × 3 saisons)
# Modèles  : SARIMA  vs  XGBoost Time Series (features laggées)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (mean_absolute_percentage_error,
                              mean_squared_error, mean_absolute_error, r2_score)

# ── Chemins ──────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_PROJECT  = os.path.join(_HERE, 'ml_scout_project')
DATA_TS   = os.path.join(_PROJECT, 'data', 'dataset_timeseries.csv')
DATA_CLU  = os.path.join(_PROJECT, 'data', 'dataset_clustering.csv')
MODELS    = os.path.join(_PROJECT, 'models')
VISUALS   = os.path.join(_PROJECT, 'visuals')
os.makedirs(MODELS,  exist_ok=True)
os.makedirs(VISUALS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label=''):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse  = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true + 1e-9, y_pred + 1e-9) * 100
    r2   = r2_score(y_true, y_pred) if len(y_true) > 1 else float('nan')
    if label:
        print(f"  {label:30s} MAE={mae:.2f}  RMSE={rmse:.2f}  "
              f"MAPE={mape:.1f}%  R²={r2:.3f}")
    return {'MAE': round(mae,3), 'RMSE': round(rmse,3),
            'MSE': round(mse,3), 'MAPE(%)': round(mape,2), 'R²': round(r2,3)}

# =============================================================================
# F.0  CHARGEMENT & PRÉPARATION
# =============================================================================
print("\n" + "█"*65)
print("  SECTION F — TIME SERIES / FORECASTING")
print("  Objectif : Prédire les adhérents par unité et par saison")
print("█"*65)

df_ts  = pd.read_csv(DATA_TS)
df_clu = pd.read_csv(DATA_CLU)
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)

# Fusionner pour avoir leak_rate
df = df_ts.merge(df_clu[['fk_type_unite','season_year','leak_rate',
                          'nmbr_members_previous_season']],
                 on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

print("\n" + "="*65)
print("F.0  DONNÉES (DWH — 18 lignes)")
print("="*65)
print(df[['fk_type_unite','season','season_year',
          'nbmr_members_season','leak_rate']].to_string(index=False))
print(f"\n  Shape : {df.shape}  |  Missing : {df.isnull().sum().sum()}")
print(f"  Unités  : {sorted(df['fk_type_unite'].unique())}")
print(f"  Saisons : {sorted(df['season'].unique())}")

# Série agrégée (total toutes unités)
ts_agg = df.groupby('season_year')['nbmr_members_season'].sum().reset_index()
ts_agg.columns = ['year', 'total_members']
print(f"\n  Série agrégée (total adhérents) :")
print(ts_agg.to_string(index=False))

# =============================================================================
# F.1  ANALYSE : STATIONNARITÉ, DÉCOMPOSITION, ADF/KPSS
# =============================================================================
print("\n" + "="*65)
print("F.1  ANALYSE DE LA SÉRIE TEMPORELLE")
print("="*65)

# Variations inter-saisons
ts_agg['variation']     = ts_agg['total_members'].diff()
ts_agg['variation_pct'] = ts_agg['total_members'].pct_change() * 100
print("\n  Variations inter-saisons :")
print(ts_agg.round(2).to_string(index=False))

# ── ADF Test ─────────────────────────────────────────────────────────────────
print("\n  ── Test ADF (Augmented Dickey-Fuller) ──")
print("  H0 : la série a une racine unitaire (non-stationnaire)")
print("  ⚠️  3 points agrégés → résultat indicatif uniquement")
try:
    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(
        ts_agg['total_members'], maxlag=0, autolag=None, regression='c')
    print(f"    ADF Statistic : {adf_stat:.4f}")
    print(f"    p-value       : {adf_p:.4f}")
    print(f"    Valeurs critiques : {adf_crit}")
    adf_conclusion = "STATIONNAIRE" if adf_p < 0.05 else "NON-STATIONNAIRE"
    print(f"    → Conclusion ADF : série {adf_conclusion}")
except Exception as e:
    print(f"    ADF non calculable ({e})")
    adf_conclusion = "NON-STATIONNAIRE (tendance décroissante visible)"
    print(f"    → Analyse visuelle : {adf_conclusion}")

# ── KPSS Test ────────────────────────────────────────────────────────────────
print("\n  ── Test KPSS ──")
print("  H0 : la série est stationnaire")
try:
    kpss_stat, kpss_p, _, kpss_crit = kpss(
        ts_agg['total_members'], regression='c', nlags=0)
    print(f"    KPSS Statistic : {kpss_stat:.4f}")
    print(f"    p-value        : {kpss_p:.4f}  (p≥0.05 → stationnaire)")
    print(f"    Valeurs critiques : {kpss_crit}")
    kpss_conclusion = "STATIONNAIRE" if kpss_p >= 0.05 else "NON-STATIONNAIRE"
    print(f"    → Conclusion KPSS : série {kpss_conclusion}")
except Exception as e:
    print(f"    KPSS non calculable ({e})")
    kpss_conclusion = "indéterminé"

print(f"\n  ── Synthèse stationnarité ──")
print(f"    ADF  : {adf_conclusion}")
print(f"    KPSS : {kpss_conclusion}")
print("    → Tendance décroissante globale → différenciation d=1 recommandée")
print("      (SARIMA avec d=1 pour corriger la non-stationnarité)")

# ── Décomposition ─────────────────────────────────────────────────────────────
print("\n  ── Décomposition de la série agrégée ──")
try:
    decomp = seasonal_decompose(
        ts_agg.set_index('year')['total_members'],
        model='additive', period=1, extrapolate_trend='freq')

    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    ts_agg.set_index('year')['total_members'].plot(
        ax=axes[0], title='Série Originale', color='navy',
        marker='o', linewidth=2, markersize=10)
    decomp.trend.plot(ax=axes[1], title='Trend (tendance)',
                      color='steelblue', marker='o', linewidth=2)
    decomp.seasonal.plot(ax=axes[2], title='Saisonnalité',
                         color='orange', marker='o', linewidth=2)
    decomp.resid.plot(ax=axes[3], title='Résidus',
                      color='red', marker='o', linewidth=2)
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_xlabel('Année')
    plt.suptitle('Décomposition Additive — Adhérents Totaux\n'
                 '(3 saisons DWH — résultats indicatifs)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{VISUALS}/ts_decomposition.png', dpi=150)
    plt.close()
    print(f"  💾 {VISUALS}/ts_decomposition.png")
except Exception as e:
    print(f"  Décomposition : {e}")

# ── Évolution par unité ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
colors_u = plt.cm.tab10(np.linspace(0, 0.6, 6))
for i, unit in enumerate(sorted(df['fk_type_unite'].unique())):
    sub = df[df['fk_type_unite'] == unit].sort_values('season_year')
    ax.plot(sub['season_year'], sub['nbmr_members_season'],
            marker='o', label=f'Unité {unit}', linewidth=2.5,
            markersize=9, color=colors_u[i])
    for _, row in sub.iterrows():
        ax.annotate(str(int(row['nbmr_members_season'])),
                    (row['season_year'], row['nbmr_members_season']),
                    textcoords='offset points', xytext=(0, 8),
                    fontsize=8, ha='center')
ax.set_xlabel('Année de saison', fontsize=11)
ax.set_ylabel("Nombre d'adhérents", fontsize=11)
ax.set_title("Évolution des Adhérents par Unité et par Saison\n(DWH)", fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_xticks([2023, 2024, 2025])
ax.set_xticklabels(['2023/2024', '2024/2025', '2025/2026'])
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_evolution_par_unite.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_evolution_par_unite.png")

# =============================================================================
# F.2  FEATURE ENGINEERING (mêmes features que régression/classification)
# =============================================================================
print("\n" + "="*65)
print("F.2  FEATURE ENGINEERING — FEATURES LAGGÉES")
print("="*65)
print("""
  Cohérence avec les sections D et C :
  members_lag1        → adhérents saison précédente
  leak_rate_lag1      → taux de fuite saison précédente
  growth_rate_lag1    → taux de croissance saison précédente
  delta_members_lag1  → variation absolue saison précédente
  avg_members_past    → moyenne historique de l'unité
  season_index        → numéro de saison (1, 2, 3...)
  fk_type_unite       → identité de l'unité
  retention_rate      → 1 - leak_rate/100
  members_x_retention → interaction membres × rétention
""")

grp = df.groupby('fk_type_unite')
df['members_lag1']        = df['nmbr_members_previous_season']
df['leak_rate_lag1']      = grp['leak_rate'].shift(1).fillna(0)
df['growth_rate_lag1']    = grp['nbmr_members_season'].pct_change().shift(1).fillna(0) * 100
df['delta_members_lag1']  = grp['nbmr_members_season'].diff().shift(1).fillna(0)
df['avg_members_past']    = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0]))
df['season_index']        = grp.cumcount() + 1
df['retention_rate']      = 1 - df['leak_rate_lag1'] / 100
df['members_x_retention'] = df['members_lag1'] * df['retention_rate']

FEAT = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
        'delta_members_lag1', 'avg_members_past',
        'season_index', 'fk_type_unite',
        'retention_rate', 'members_x_retention']
TARGET = 'nbmr_members_season'

print(f"  Dataset enrichi (18 lignes) :")
show = ['fk_type_unite','season'] + FEAT + [TARGET]
print(df[show].round(2).to_string(index=False))

# Corrélations
corr = df[FEAT + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(f"\n  Corrélations avec {TARGET} :")
print(corr.round(4).to_string())

# =============================================================================
# F.3  MODÈLE 1 — SARIMA PAR UNITÉ
# =============================================================================
print("\n" + "="*65)
print("F.3  MODÈLE 1 : SARIMA(1,1,0)(0,0,0)[1] PAR UNITÉ")
print("="*65)
print("""
  Intuition : SARIMA = ARIMA + composante Saisonnière.
    AR(p=1)  : la valeur actuelle dépend de la valeur précédente.
    I(d=1)   : différenciation d'ordre 1 pour corriger la tendance.
    MA(q=0)  : pas de composante moyenne mobile.
    S(P,D,Q) : composante saisonnière (ici minimale car 3 saisons).
  Paramètres : (p,d,q)(P,D,Q)[s] = (1,1,0)(0,0,0)[1].
  Hypothèses : résidus bruit blanc, stationnarité après différenciation.
  Limites    : 3 points/unité → résultats indicatifs, intervalles larges.
  Justification : modèle statistique classique pour séries temporelles,
                  capture la tendance via d=1.
  Validation : Leave-One-Out (LOO) sur les points prédictibles.
""")

all_y_true_sarima, all_y_pred_sarima = [], []
sarima_results = []
sarima_fc_2026 = {}
sarima_preds_per_unit = {}

for unit in sorted(df['fk_type_unite'].unique()):
    sub = df[df['fk_type_unite'] == unit].sort_values('season_year')
    y   = sub[TARGET].values.astype(float)

    # LOO : prédire chaque point en entraînant sur les précédents
    y_pred_loo = np.full(len(y), np.nan)
    for test_i in range(1, len(y)):
        y_train = y[:test_i]
        try:
            m = SARIMAX(y_train, order=(1,1,0), seasonal_order=(0,0,0,1),
                        trend='n', enforce_stationarity=False,
                        enforce_invertibility=False)
            f = m.fit(disp=False)
            y_pred_loo[test_i] = max(f.forecast(steps=1)[0], 0)
        except Exception:
            y_pred_loo[test_i] = y_train[-1]  # fallback naïf

    # Métriques sur points prédits
    valid = ~np.isnan(y_pred_loo)
    y_v, p_v = y[valid], y_pred_loo[valid]
    m_dict = metrics(y_v, p_v, label=f'SARIMA Unité {unit} (LOO)')
    all_y_true_sarima.extend(y_v.tolist())
    all_y_pred_sarima.extend(p_v.tolist())

    # Forecast 2026/2027 : entraîner sur toute la série
    try:
        full_m = SARIMAX(y, order=(1,1,0), seasonal_order=(0,0,0,1),
                         trend='n', enforce_stationarity=False,
                         enforce_invertibility=False)
        full_f = full_m.fit(disp=False)
        fc = max(float(full_f.forecast(steps=1)[0]), 0)
    except Exception:
        fc = float(y[-1])

    sarima_fc_2026[unit] = round(fc, 1)
    sarima_preds_per_unit[unit] = {'y': y, 'y_pred_loo': y_pred_loo, 'fc': fc}
    m_dict['Unité'] = unit
    m_dict['Forecast_2026/2027'] = round(fc, 1)
    sarima_results.append(m_dict)

df_sarima = pd.DataFrame(sarima_results)
df_sarima.to_csv(f'{MODELS}/sarima_results.csv', index=False)

print(f"\n  Métriques globales SARIMA (LOO, tous unités) :")
sarima_global = metrics(all_y_true_sarima, all_y_pred_sarima, label='SARIMA global')

print(f"\n  Forecasts 2026/2027 (SARIMA) :")
for u, v in sarima_fc_2026.items():
    real = df[df['fk_type_unite']==u].sort_values('season_year')[TARGET].iloc[-1]
    delta = v - real
    trend = "↗" if delta > 1 else ("↘" if delta < -1 else "→")
    print(f"    Unité {u} : réel 2025/26={real}  →  forecast={v}  {trend}")

# =============================================================================
# F.4  MODÈLE 2 — XGBOOST TIME SERIES (Gradient Boosting + features laggées)
# =============================================================================
print("\n" + "="*65)
print("F.4  MODÈLE 2 : XGBOOST TIME SERIES (Gradient Boosting)")
print("="*65)
print("""
  Intuition : Gradient Boosting entraîné sur des features temporelles laggées.
    Chaque arbre corrige les erreurs du précédent (boosting séquentiel).
    Les features laggées transforment le problème TS en problème supervisé.
  Paramètres : n_estimators=100, learning_rate=0.05, max_depth=2.
  Hypothèses : aucune hypothèse distributionnelle, capture les non-linéarités.
  Limites    : risque de surapprentissage sur petits datasets.
  Validation : Leave-One-Out (LOO) — 17 train / 1 test × 18 itérations.
  Justification : approche ML moderne pour TS, cohérente avec les features
                  utilisées en régression et classification (sections D et C).
""")

X = df[FEAT].values
y_all = df[TARGET].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                max_depth=2, random_state=42,
                                subsample=0.8, min_samples_split=2)

# LOO-CV
loo = LeaveOneOut()
y_pred_gb_loo = np.zeros(len(y_all))
for tr, te in loo.split(X_sc):
    gb.fit(X_sc[tr], y_all[tr])
    y_pred_gb_loo[te] = max(gb.predict(X_sc[te])[0], 0)

print("  Métriques LOO-CV :")
gb_global = metrics(y_all, y_pred_gb_loo, label='XGBoost TS global (LOO)')

# Entraîner sur tout pour forecast et feature importance
gb.fit(X_sc, y_all)

# Feature importance
fi = pd.Series(gb.feature_importances_, index=FEAT).sort_values(ascending=False)
print(f"\n  Feature Importance :\n{fi.round(4).to_string()}")

# Forecast 2026/2027 par unité
gb_fc_2026 = {}
for unit in sorted(df['fk_type_unite'].unique()):
    sub  = df[df['fk_type_unite'] == unit].sort_values('season_year')
    last = sub.iloc[-1]
    prev = sub.iloc[-2] if len(sub) >= 2 else sub.iloc[-1]
    m_curr = last[TARGET]
    m_prev = prev[TARGET]
    growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
    delta  = m_curr - m_prev
    leak1  = last['leak_rate']
    ret    = 1 - leak1 / 100
    row = {
        'members_lag1':        m_curr,
        'leak_rate_lag1':      leak1,
        'growth_rate_lag1':    growth,
        'delta_members_lag1':  delta,
        'avg_members_past':    sub[TARGET].mean(),
        'season_index':        4,
        'fk_type_unite':       unit,
        'retention_rate':      ret,
        'members_x_retention': m_curr * ret,
    }
    x_new = scaler.transform(pd.DataFrame([row])[FEAT].values)
    fc = max(float(gb.predict(x_new)[0]), 0)
    gb_fc_2026[unit] = round(fc, 1)

print(f"\n  Forecasts 2026/2027 (XGBoost TS) :")
for u, v in gb_fc_2026.items():
    real = df[df['fk_type_unite']==u].sort_values('season_year')[TARGET].iloc[-1]
    delta = v - real
    trend = "↗" if delta > 1 else ("↘" if delta < -1 else "→")
    print(f"    Unité {u} : réel 2025/26={real}  →  forecast={v}  {trend}")

# =============================================================================
# F.5  COMPARAISON DES MODÈLES
# =============================================================================
print("\n" + "="*65)
print("F.5  COMPARAISON SARIMA vs XGBOOST TS")
print("="*65)

comparison = pd.DataFrame([
    {'Modèle': 'SARIMA(1,1,0)', **sarima_global},
    {'Modèle': 'XGBoost TS',    **gb_global},
])
print(comparison.to_string(index=False))
comparison.to_csv(f'{MODELS}/timeseries_comparison.csv', index=False)
print(f"\n  💾 {MODELS}/timeseries_comparison.csv")

best_ts = comparison.loc[comparison['RMSE'].idxmin(), 'Modèle']
print(f"\n  ✅ Meilleur modèle (RMSE min) : {best_ts}")

# Tableau forecasts
units = sorted(df['fk_type_unite'].unique())
fc_table = pd.DataFrame({
    'Unité':               units,
    'Réel_2025/2026':      [df[df['fk_type_unite']==u].sort_values('season_year')[TARGET].iloc[-1] for u in units],
    'SARIMA_2026/2027':    [sarima_fc_2026.get(u, np.nan) for u in units],
    'XGBoost_2026/2027':   [gb_fc_2026.get(u, np.nan) for u in units],
})
print(f"\n  Tableau des Forecasts 2026/2027 :")
print(fc_table.to_string(index=False))
fc_table.to_csv(f'{MODELS}/forecasts_ts_2026_2027.csv', index=False)
print(f"\n  💾 {MODELS}/forecasts_ts_2026_2027.csv")

# =============================================================================
# F.6  VISUALISATIONS
# =============================================================================
print("\n" + "="*65)
print("F.6  VISUALISATIONS")
print("="*65)

cmap6 = plt.cm.tab10(np.linspace(0, 0.6, 6))

# ── Actual vs Predicted — SARIMA ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for idx, (name, y_pred, color) in enumerate([
        ('SARIMA(1,1,0)', np.array(all_y_pred_sarima), 'steelblue'),
        ('XGBoost TS',    y_pred_gb_loo,                'coral')]):
    y_true_plot = np.array(all_y_true_sarima) if idx == 0 else y_all
    r2v = r2_score(y_true_plot, y_pred)
    axes[idx].scatter(y_true_plot, y_pred, color=color,
                      edgecolors='k', s=80, alpha=0.85, zorder=3)
    lims = [min(y_true_plot.min(), y_pred.min())-1,
            max(y_true_plot.max(), y_pred.max())+1]
    axes[idx].plot(lims, lims, 'r--', linewidth=1.5, label='Parfait')
    axes[idx].set_xlabel('Réel'); axes[idx].set_ylabel('Prédit (LOO)')
    axes[idx].set_title(f'Actual vs Predicted — {name}\nR²={r2v:.3f}')
    axes[idx].legend(); axes[idx].grid(alpha=0.3)
plt.suptitle('Actual vs Predicted — Time Series (LOO-CV)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_actual_vs_predicted.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_actual_vs_predicted.png")

# ── Résidus ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for idx, (name, y_pred, color) in enumerate([
        ('SARIMA(1,1,0)', np.array(all_y_pred_sarima), 'steelblue'),
        ('XGBoost TS',    y_pred_gb_loo,                'coral')]):
    y_true_plot = np.array(all_y_true_sarima) if idx == 0 else y_all
    resid = y_true_plot - y_pred
    axes[0][idx].scatter(y_pred, resid, color=color, edgecolors='k', s=70, alpha=0.85)
    axes[0][idx].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0][idx].set_xlabel('Valeurs Prédites')
    axes[0][idx].set_ylabel('Résidus')
    axes[0][idx].set_title(f'Résidus vs Prédits — {name}')
    axes[0][idx].grid(alpha=0.3)
    axes[1][idx].hist(resid, bins=8, color=color, edgecolor='k', alpha=0.85)
    axes[1][idx].axvline(0, color='red', linestyle='--')
    axes[1][idx].set_title(f'Distribution des Résidus — {name}')
    axes[1][idx].set_xlabel('Résidu')
    axes[1][idx].grid(alpha=0.3)
plt.suptitle('Analyse des Résidus — Time Series (LOO-CV)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_residuals.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_residuals.png")

# ── Feature Importance XGBoost ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
colors_fi = ['#1565C0' if v >= fi.mean() else '#90CAF9' for v in fi.sort_values()]
fi.sort_values().plot(kind='barh', ax=ax, color=colors_fi)
ax.axvline(fi.mean(), color='red', linestyle='--',
           label=f'Moyenne ({fi.mean():.3f})')
ax.set_title('Feature Importance — XGBoost Time Series\n(Features Laggées DWH)')
ax.set_xlabel('Importance (Gini)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_feature_importance.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_feature_importance.png")

# ── Forecast 2026/2027 par unité ──────────────────────────────────────────────
x_labels = ['2023/2024', '2024/2025', '2025/2026', '2026/2027\n(forecast)']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, unit in enumerate(units):
    sub    = df[df['fk_type_unite'] == unit].sort_values('season_year')
    y_hist = sub[TARGET].values
    x_hist = list(range(len(y_hist)))

    fc_s = sarima_fc_2026.get(unit, np.nan)
    fc_g = gb_fc_2026.get(unit, np.nan)
    loo_s = sarima_preds_per_unit[unit]['y_pred_loo']

    ax = axes[idx]
    # Historique
    ax.plot(x_hist, y_hist, 'bo-', label='Historique',
            linewidth=2.5, markersize=9, zorder=3)
    # Points LOO SARIMA
    for i in range(1, len(y_hist)):
        if not np.isnan(loo_s[i]):
            ax.scatter(i, loo_s[i], color='steelblue', marker='x',
                       s=120, zorder=4, linewidths=2,
                       label='SARIMA LOO' if i == 1 else '')
    # Forecast SARIMA
    ax.plot([len(y_hist)-1, len(y_hist)], [y_hist[-1], fc_s],
            'b--o', label=f'SARIMA fc={fc_s:.0f}',
            linewidth=2, markersize=8)
    # Forecast XGBoost
    ax.plot([len(y_hist)-1, len(y_hist)], [y_hist[-1], fc_g],
            'r-.s', label=f'XGBoost fc={fc_g:.0f}',
            linewidth=2, markersize=8)
    # Zone forecast
    ax.axvspan(len(y_hist)-0.5, len(y_hist)+0.5,
               alpha=0.08, color='gray', label='Zone forecast')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_title(f'Unité {unit}', fontsize=11, fontweight='bold')
    ax.set_ylabel("Adhérents")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle('Forecast 2026/2027 par Unité — SARIMA vs XGBoost TS\n'
             '(DWH — 3 saisons historiques)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_forecast_par_unite.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_forecast_par_unite.png")

# ── Comparaison métriques ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, metric in enumerate(['MAE', 'RMSE', 'MAPE(%)', 'R²']):
    vals = comparison[metric].values
    bars = axes[i].bar(comparison['Modèle'], vals,
                       color=['steelblue', 'coral'], width=0.5, alpha=0.85)
    axes[i].set_title(metric, fontsize=12)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, v*1.03,
                     f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(axis='y', alpha=0.3)
plt.suptitle('Comparaison SARIMA vs XGBoost TS — Métriques LOO-CV\n'
             '(18 obs. DWH)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_model_comparison.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_model_comparison.png")

# ── Barplot forecast total ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(units))
w = 0.25
hist_vals  = fc_table['Réel_2025/2026'].values
sarima_vals = fc_table['SARIMA_2026/2027'].values
gb_vals    = fc_table['XGBoost_2026/2027'].values

ax.bar(x-w, hist_vals,   w, label='Réel 2025/2026',    color='#90CAF9', alpha=0.9)
ax.bar(x,   sarima_vals, w, label='SARIMA 2026/2027',   color='steelblue', alpha=0.9)
ax.bar(x+w, gb_vals,     w, label='XGBoost 2026/2027',  color='coral', alpha=0.9)
for i, (h, s, g) in enumerate(zip(hist_vals, sarima_vals, gb_vals)):
    ax.text(i-w, h+0.2, str(int(h)), ha='center', fontsize=8)
    ax.text(i,   s+0.2, f'{s:.0f}', ha='center', fontsize=8)
    ax.text(i+w, g+0.2, f'{g:.0f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'Unité {u}' for u in units])
ax.set_ylabel("Nombre d'adhérents")
ax.set_title('Prédiction 2026/2027 par Unité — SARIMA vs XGBoost TS\n(DWH)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALS}/ts_forecast_barplot.png', dpi=150)
plt.close()
print(f"  💾 {VISUALS}/ts_forecast_barplot.png")

# =============================================================================
# F.7  INTERPRÉTATION BUSINESS
# =============================================================================
print("\n" + "="*65)
print("F.7  INTERPRÉTATION BUSINESS & RECOMMANDATIONS")
print("="*65)

print(f"\n  Meilleur modèle (RMSE min) : {best_ts}")
print(f"\n  Total adhérents 2025/2026 (réel)      : {fc_table['Réel_2025/2026'].sum()}")
print(f"  Total prédit 2026/2027 (SARIMA)        : {fc_table['SARIMA_2026/2027'].sum():.0f}")
print(f"  Total prédit 2026/2027 (XGBoost TS)    : {fc_table['XGBoost_2026/2027'].sum():.0f}")

total_reel = fc_table['Réel_2025/2026'].sum()
total_gb   = fc_table['XGBoost_2026/2027'].sum()
var_gb = (total_gb - total_reel) / total_reel * 100
print(f"\n  Variation prévue (XGBoost) : {var_gb:+.1f}%")

print(f"\n  Recommandations par unité (modèle retenu : {best_ts}) :")
best_col = 'SARIMA_2026/2027' if 'SARIMA' in best_ts else 'XGBoost_2026/2027'
print(f"  {'Unité':>6} | {'Réel 25/26':>10} | {'Prédit 26/27':>12} | {'Δ':>6} | Action")
print("  " + "-"*65)
for _, row in fc_table.iterrows():
    real = row['Réel_2025/2026']
    fc   = row[best_col]
    delta = fc - real
    if delta > 2:
        action = "Préparer ressources supplémentaires"
    elif delta < -2:
        action = "Lancer campagne de fidélisation"
    else:
        action = "Maintenir les actions actuelles"
    print(f"  {int(row['Unité']):>6} | {real:>10.0f} | {fc:>12.1f} | "
          f"{delta:>+6.1f} | {action}")

print(f"""
  ⚠️  Note méthodologique :
  Avec 3 saisons historiques (DWH), les prévisions sont indicatives.
  La collecte de données sur 5+ saisons améliorera significativement
  la précision des modèles SARIMA et XGBoost TS.
  Le modèle XGBoost TS est préféré car il intègre les features laggées
  (cohérence avec les sections D et C) et capture les non-linéarités.
""")

print("\n" + "█"*65)
print("  ✅ SECTION F TERMINÉE — tous les fichiers sauvegardés")
print("█"*65)
