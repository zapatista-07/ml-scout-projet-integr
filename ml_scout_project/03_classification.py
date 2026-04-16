# =============================================================================
# 03_classification.py — Section C : Classification (≥ 2 modèles)
# Objectif : Prédire dropout_risk avec features laggées enrichies
# Dataset  : 18 observations DWH (toutes les lignes utilisées)
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

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

os.makedirs('models',  exist_ok=True)
os.makedirs('visuals', exist_ok=True)

FEATURE_COLS = ['members_lag1', 'leak_rate_lag1', 'growth_rate_lag1',
                'delta_members_lag1', 'avg_members_unite_past',
                'season_index', 'fk_type_unite',
                'retention_rate', 'members_x_retention']
TARGET = 'dropout_risk'

# ─────────────────────────────────────────────────────────────────────────────
# C.0  CHARGEMENT & FEATURE ENGINEERING (18 lignes DWH)
# ─────────────────────────────────────────────────────────────────────────────
df_cls = pd.read_csv('data/dataset_classification.csv')
df_clu = pd.read_csv('data/dataset_clustering.csv')
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)

df = df_cls.merge(df_clu[['fk_type_unite','season_year','leak_rate']],
                  on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

# Feature Engineering — mêmes features que la régression
grp = df.groupby('fk_type_unite')
df['members_lag1']           = df['nmbr_members_previous_season']
df['leak_rate_lag1']         = grp['leak_rate'].shift(1).fillna(0)
df['growth_rate_lag1']       = grp['nbmr_members_season'].pct_change().shift(1).fillna(0) * 100
df['delta_members_lag1']     = grp['nbmr_members_season'].diff().shift(1).fillna(0)
df['avg_members_unite_past'] = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0]))
df['season_index']           = grp.cumcount() + 1
df['retention_rate']         = 1 - df['leak_rate_lag1'] / 100
df['members_x_retention']    = df['members_lag1'] * df['retention_rate']

X = df[FEATURE_COLS]
y = df[TARGET].astype(int)

print("="*65)
print("C.0  DONNÉES & FEATURE ENGINEERING (DWH — 18 lignes)")
print("="*65)
print(f"  Lignes : {len(df)}  |  Missing : {X.isnull().sum().sum()}")
show = ['fk_type_unite','season_year'] + FEATURE_COLS + [TARGET]
print(df[show].round(2).to_string(index=False))
print(f"\n  Distribution cible :\n{y.value_counts().to_string()}")
print(f"  Ratio : {y.value_counts()[0]}/{y.value_counts()[1]} (0=no dropout / 1=dropout)")

print("\n  Corrélations avec dropout_risk :")
corr = df[FEATURE_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(corr.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# C.1  EXPLICATION DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("C.1  EXPLICATION DES MODÈLES")
print("="*65)
print("""
Approche : Features Laggées + Feature Engineering
  "Pas de nouvelles données — uniquement des indicateurs dérivés du DWH."
  members_lag1        → membres saison précédente
  leak_rate_lag1      → taux de fuite saison précédente
  growth_rate_lag1    → taux de croissance saison précédente
  delta_members_lag1  → variation absolue saison précédente
  avg_members_unite_past → moyenne historique de l'unité
  season_index        → numéro de saison
  fk_type_unite       → identité de l'unité
  retention_rate      → 1 - leak_rate/100 (taux de fidélisation)
  members_x_retention → interaction membres × rétention (corrélation 0.79 !)

Modèle 1 — Random Forest Classifier
  Intuition  : ensemble d'arbres de décision (bagging). Chaque arbre vote,
               la classe majoritaire est retenue. Réduit la variance.
  Paramètres : n_estimators, max_depth, class_weight='balanced' (GridSearch).
  Hypothèses : aucune hypothèse distributionnelle.
  Limites    : boîte noire, risque de surapprentissage sur petits datasets.
  Justification : robuste, gère le déséquilibre, fournit feature importance.

Modèle 2 — Logistic Regression
  Intuition  : modèle linéaire estimant P(dropout=1) via la sigmoïde.
               log(p/1-p) = β₀ + β₁x₁ + ... + βₙxₙ
  Paramètres : C (régularisation inverse), solver, class_weight='balanced'.
  Hypothèses : linéarité du log-odds, indépendance des observations.
  Limites    : ne capture pas les interactions complexes.
  Justification : interprétable, coefficients explicables, bon baseline.

  ⚠️  Validation : Leave-One-Out CV (18 obs. → 17 train / 1 test × 18 iter.)
""")

# ─────────────────────────────────────────────────────────────────────────────
# C.2  PIPELINES + GRIDSEARCH (LOO-CV)
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("C.2  PIPELINES + GRIDSEARCH (LOO-CV)")
print("="*65)

loo = LeaveOneOut()

# Random Forest
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(random_state=42, class_weight='balanced'))
])
rf_gs = GridSearchCV(rf_pipe, {
    'model__n_estimators':     [50, 100, 200],
    'model__max_depth':        [2, 3, 4],
    'model__min_samples_split':[2, 3],
    'model__max_features':     ['sqrt', 0.7, 0.9],
}, cv=loo, scoring='f1', n_jobs=-1)
rf_gs.fit(X, y)
best_rf = rf_gs.best_estimator_
print(f"\n  RF — meilleurs params : {rf_gs.best_params_}"
      f"  (F1 LOO : {rf_gs.best_score_:.3f})")

# Logistic Regression
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(random_state=42, max_iter=2000,
                                  class_weight='balanced'))
])
lr_gs = GridSearchCV(lr_pipe, {
    'model__C':      [0.001, 0.01, 0.1, 1, 10, 100],
    'model__solver': ['lbfgs', 'liblinear'],
    'model__penalty':['l2'],
}, cv=loo, scoring='f1', n_jobs=-1)
lr_gs.fit(X, y)
best_lr = lr_gs.best_estimator_
print(f"  LR — meilleurs params : {lr_gs.best_params_}"
      f"  (F1 LOO : {lr_gs.best_score_:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# C.3  VALIDATION LOO-CV — MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("C.3  VALIDATION LOO-CV — MÉTRIQUES")
print("="*65)
print("  Justification : 18 obs. → LOO = 17 train / 1 test × 18 itérations.\n")

results_rows = []
all_preds, all_probas = {}, {}

for name, pipe in [('Random Forest', best_rf), ('Logistic Regression', best_lr)]:
    y_pred = np.zeros(len(y), dtype=int)
    y_prob = np.zeros(len(y))
    for tr, te in loo.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        y_pred[te] = pipe.predict(X.iloc[te])
        y_prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    all_preds[name]  = y_pred
    all_probas[name] = y_prob

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    try:    auc = roc_auc_score(y, y_prob)
    except: auc = float('nan')

    results_rows.append({'Modèle': name, 'Accuracy': round(acc,3),
                         'Precision': round(prec,3), 'Recall': round(rec,3),
                         'F1-Score': round(f1,3), 'ROC-AUC': round(auc,3)})
    print(f"  {name} :")
    print(f"    Accuracy={acc:.3f}  Precision={prec:.3f}  "
          f"Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print(f"\n  Rapport détaillé :\n{classification_report(y, y_pred, zero_division=0)}")

# Ré-entraîner sur tout pour visualisations
for name, pipe in [('Random Forest', best_rf), ('Logistic Regression', best_lr)]:
    pipe.fit(X, y)

results_df = pd.DataFrame(results_rows)
results_df.to_csv('models/classification_comparison.csv', index=False)
best_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Modèle']
print(f"  ✅ Meilleur modèle (F1 LOO max) : {best_name}")

# ─────────────────────────────────────────────────────────────────────────────
# C.4  PRÉDICTION DROPOUT 2026/2027
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("C.4  PRÉDICTION DROPOUT 2026/2027")
print("="*65)

df_future = pd.read_csv('models/predictions_2026_2027.csv')

# Construire les features pour 2026/2027
pred_cls_rows = []
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

    pred_cls_rows.append({
        'fk_type_unite':          unit,
        'members_lag1':           m_curr,
        'leak_rate_lag1':         leak1,
        'growth_rate_lag1':       growth,
        'delta_members_lag1':     delta,
        'avg_members_unite_past': hist['nbmr_members_season'].mean(),
        'season_index':           4,
        'retention_rate':         ret,
        'members_x_retention':    m_curr * ret,
    })

df_cls_future = pd.DataFrame(pred_cls_rows)
X_future = df_cls_future[FEATURE_COLS]

best_model = best_rf if best_name == 'Random Forest' else best_lr
df_cls_future['dropout_risk_pred'] = best_model.predict(X_future)
df_cls_future['dropout_proba']     = best_model.predict_proba(X_future)[:, 1].round(3)

print(f"\n  {'Unité':>6} | {'P(dropout)':>10} | {'Prédiction':>12} | {'Risque'}")
print("  " + "-"*50)
for _, row in df_cls_future.iterrows():
    risk = "🔴 DROPOUT" if row['dropout_risk_pred'] == 1 else "🟢 OK"
    print(f"  {int(row['fk_type_unite']):>6} | {row['dropout_proba']:>10.3f} | "
          f"{'Dropout' if row['dropout_risk_pred']==1 else 'No dropout':>12} | {risk}")

df_cls_future.to_csv('models/predictions_dropout_2026.csv', index=False)
print(f"\n  💾 models/predictions_dropout_2026.csv")

# ─────────────────────────────────────────────────────────────────────────────
# C.5  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (name, yp) in enumerate(all_preds.items()):
    cm = confusion_matrix(y, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No dropout','Dropout'],
                yticklabels=['No dropout','Dropout'])
    f1v = f1_score(y, yp, zero_division=0)
    auc = roc_auc_score(y, all_probas[name]) if len(np.unique(y)) > 1 else float('nan')
    axes[idx].set_title(f'{name}\nAcc={accuracy_score(y,yp):.3f}  '
                        f'F1={f1v:.3f}  AUC={auc:.3f}')
    axes[idx].set_xlabel('Prédit'); axes[idx].set_ylabel('Réel')
plt.suptitle('Matrices de Confusion — Classification (LOO-CV)\n'
             '(Features Laggées, 18 obs. DWH)', fontsize=13)
plt.tight_layout()
plt.savefig('visuals/confusion_matrix_classification.png', dpi=150)
plt.close()
print("\n  💾 visuals/confusion_matrix_classification.png")

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for (name, yprb), color in zip(all_probas.items(), ['steelblue', 'coral']):
    try:
        fpr, tpr, _ = roc_curve(y, yprb)
        auc = roc_auc_score(y, yprb)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC={auc:.3f})')
    except Exception:
        pass
ax.fill_between([0,1],[0,1], alpha=0.08, color='gray')
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Aléatoire (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Courbes ROC — Classification (LOO-CV)\n(Features Laggées, 18 obs. DWH)')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/roc_curve_classification.png', dpi=150)
plt.close()
print("  💾 visuals/roc_curve_classification.png")

# Feature importance RF
rf_obj = best_rf.named_steps['model']
fi = pd.Series(rf_obj.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
colors_fi = ['#1565C0' if v >= fi.mean() else '#90CAF9' for v in fi.sort_values()]
fi.sort_values().plot(kind='barh', ax=ax, color=colors_fi)
ax.axvline(fi.mean(), color='red', linestyle='--',
           label=f'Moyenne ({fi.mean():.3f})')
ax.set_title('Feature Importance — Random Forest Classifier\n(Features Laggées DWH)')
ax.set_xlabel('Importance (Gini)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/feature_importance_classification.png', dpi=150)
plt.close()
print("  💾 visuals/feature_importance_classification.png")
print(f"\n  Feature Importance :\n{fi.round(4).to_string()}")

# Comparaison métriques
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, metric in enumerate(['Accuracy','Precision','Recall','F1-Score','ROC-AUC']):
    vals = results_df[metric].values
    bars = axes[i].bar(results_df['Modèle'], vals,
                       color=['steelblue','coral'], width=0.5, alpha=0.85)
    axes[i].set_title(metric); axes[i].set_ylim(0, 1.15)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, v+0.02,
                     f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(axis='y', alpha=0.3)
plt.suptitle('Comparaison RF vs Logistic Regression — Métriques LOO-CV\n'
             '(Features Laggées, 18 obs. DWH)', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/classification_metrics_comparison.png', dpi=150)
plt.close()
print("  💾 visuals/classification_metrics_comparison.png")

# Probabilités dropout 2026/2027
fig, ax = plt.subplots(figsize=(9, 4))
units = df_cls_future['fk_type_unite'].astype(int)
probas = df_cls_future['dropout_proba'].values
colors_bar = ['coral' if p >= 0.5 else 'steelblue' for p in probas]
bars = ax.bar([f'Unité {u}' for u in units], probas, color=colors_bar, alpha=0.85)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Seuil 0.5')
for bar, v in zip(bars, probas):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
            f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('P(dropout)')
ax.set_title('Probabilité de Dropout 2026/2027 par Unité\n'
             '(Rouge = risque élevé, Bleu = faible risque)')
ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/dropout_proba_2026.png', dpi=150)
plt.close()
print("  💾 visuals/dropout_proba_2026.png")

joblib.dump(best_rf, 'models/classifier_rf.pkl')
joblib.dump(best_lr, 'models/classifier_lr.pkl')

print("\n" + "="*65)
print("✅ CLASSIFICATION TERMINÉE")
print("="*65)
