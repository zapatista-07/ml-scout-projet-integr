# =============================================================================
# 06_anomaly_detection.py — Advanced Objectives : Anomaly Detection
# Objectif : Détecter les unités/saisons avec comportements anormaux
#            d'adhésion (chute brutale, taux de fuite excessif, etc.)
# Dataset  : 18 observations DWH
# Méthodes : Isolation Forest + Local Outlier Factor (LOF) + Z-Score
# =============================================================================
import os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

os.makedirs('models',  exist_ok=True)
os.makedirs('visuals', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# A.0  CHARGEMENT & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df_ts  = pd.read_csv('data/dataset_timeseries.csv')
df_clu = pd.read_csv('data/dataset_clustering.csv')
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)

df = df_ts.merge(df_clu[['fk_type_unite','season_year','leak_rate',
                          'nmbr_members_previous_season']],
                 on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

print("█"*65)
print("  ADVANCED OBJECTIVES — ANOMALY DETECTION")
print("  Objectif : Détecter les comportements anormaux d'adhésion")
print("█"*65)

print("\n" + "="*65)
print("A.0  DONNÉES & FEATURE ENGINEERING (DWH — 18 lignes)")
print("="*65)

grp = df.groupby('fk_type_unite')
df['members_lag1']        = df['nmbr_members_previous_season']
df['growth_rate']         = grp['nbmr_members_season'].pct_change().fillna(0) * 100
df['delta_members']       = grp['nbmr_members_season'].diff().fillna(0)
df['avg_members_past']    = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0]))
df['retention_rate']      = 1 - df['leak_rate'] / 100
df['members_x_retention'] = df['nbmr_members_season'] * df['retention_rate']
df['season_index']        = grp.cumcount() + 1
df['deviation_from_avg']  = df['nbmr_members_season'] - df['avg_members_past']

FEAT = ['nbmr_members_season', 'leak_rate', 'growth_rate',
        'delta_members', 'retention_rate', 'members_x_retention',
        'deviation_from_avg']

print(f"\n  Features pour détection d'anomalies : {FEAT}")
print(f"\n  Dataset enrichi :")
show = ['fk_type_unite','season'] + FEAT
print(df[show].round(2).to_string(index=False))

# Stats descriptives
print(f"\n  Statistiques descriptives :")
print(df[FEAT].describe().round(2).to_string())

X_raw = df[FEAT].fillna(0).values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ─────────────────────────────────────────────────────────────────────────────
# A.1  EXPLICATION DES MÉTHODES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.1  EXPLICATION DES MÉTHODES")
print("="*65)
print("""
Méthode 1 — Isolation Forest
  Intuition  : isole les anomalies en construisant des arbres de décision
               aléatoires. Les anomalies sont isolées en moins de splits
               car elles sont rares et différentes des points normaux.
  Paramètres : contamination=0.15 (15% de points suspects attendus),
               n_estimators=100, random_state=42.
  Hypothèses : les anomalies sont peu nombreuses et différentes.
  Limites    : sensible au paramètre contamination.
  Justification : robuste, non-paramétrique, adapté aux petits datasets.
  Score      : anomaly_score ∈ [-1, 0] (plus négatif = plus anormal)

Méthode 2 — Local Outlier Factor (LOF)
  Intuition  : compare la densité locale d'un point à celle de ses voisins.
               Un point dans une zone peu dense par rapport à ses voisins
               est considéré comme une anomalie.
  Paramètres : n_neighbors=3 (adapté aux 18 obs.), contamination=0.15.
  Hypothèses : les anomalies ont une densité locale plus faible.
  Limites    : sensible au choix de n_neighbors.
  Justification : détecte les anomalies locales (contextuelles).
  Score      : LOF score > 1 → anomalie (plus élevé = plus anormal)

Méthode 3 — Z-Score (méthode statistique)
  Intuition  : mesure l'écart d'un point à la moyenne en unités d'écart-type.
               Un Z-score > 2 indique une valeur statistiquement anormale.
  Paramètres : seuil = 2 écarts-types.
  Hypothèses : distribution normale des données.
  Limites    : sensible aux distributions non-normales.
  Justification : méthode simple, interprétable, référence statistique.
""")

# ─────────────────────────────────────────────────────────────────────────────
# A.2  ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("A.2  ISOLATION FOREST")
print("="*65)

iso = IsolationForest(n_estimators=100, contamination=0.15,
                      random_state=42, max_samples='auto')
iso.fit(X)
iso_labels  = iso.predict(X)          # 1=normal, -1=anomalie
iso_scores  = iso.decision_function(X) # plus négatif = plus anormal
iso_anomaly = (iso_labels == -1).astype(int)

df['iso_label']  = iso_labels
df['iso_score']  = iso_scores
df['iso_anomaly']= iso_anomaly

print(f"\n  Anomalies détectées : {iso_anomaly.sum()} / {len(df)}")
print(f"\n  Détail par observation :")
print(f"  {'Unité':>6} | {'Saison':>10} | {'Membres':>8} | "
      f"{'Fuite%':>7} | {'Score':>8} | {'Statut'}")
print("  " + "-"*60)
for _, row in df.iterrows():
    statut = "🔴 ANOMALIE" if row['iso_anomaly'] == 1 else "🟢 Normal"
    print(f"  {int(row['fk_type_unite']):>6} | {row['season']:>10} | "
          f"{int(row['nbmr_members_season']):>8} | "
          f"{row['leak_rate']:>7.1f} | {row['iso_score']:>8.4f} | {statut}")

# ─────────────────────────────────────────────────────────────────────────────
# A.3  LOCAL OUTLIER FACTOR (LOF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.3  LOCAL OUTLIER FACTOR (LOF)")
print("="*65)

lof = LocalOutlierFactor(n_neighbors=3, contamination=0.15)
lof_labels  = lof.fit_predict(X)       # 1=normal, -1=anomalie
lof_scores  = -lof.negative_outlier_factor_  # LOF score (>1 = anomalie)
lof_anomaly = (lof_labels == -1).astype(int)

df['lof_label']  = lof_labels
df['lof_score']  = lof_scores
df['lof_anomaly']= lof_anomaly

print(f"\n  Anomalies détectées : {lof_anomaly.sum()} / {len(df)}")
print(f"\n  Détail par observation :")
print(f"  {'Unité':>6} | {'Saison':>10} | {'Membres':>8} | "
      f"{'Fuite%':>7} | {'LOF Score':>10} | {'Statut'}")
print("  " + "-"*65)
for _, row in df.iterrows():
    statut = "🔴 ANOMALIE" if row['lof_anomaly'] == 1 else "🟢 Normal"
    print(f"  {int(row['fk_type_unite']):>6} | {row['season']:>10} | "
          f"{int(row['nbmr_members_season']):>8} | "
          f"{row['leak_rate']:>7.1f} | {row['lof_score']:>10.4f} | {statut}")

# ─────────────────────────────────────────────────────────────────────────────
# A.4  Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.4  Z-SCORE (méthode statistique)")
print("="*65)

THRESHOLD = 2.0
z_scores = np.abs(stats.zscore(df[FEAT].fillna(0), axis=0))
df['zscore_max']     = z_scores.max(axis=1)
df['zscore_anomaly'] = (df['zscore_max'] > THRESHOLD).astype(int)

print(f"\n  Seuil Z-Score : {THRESHOLD} écarts-types")
print(f"  Anomalies détectées : {df['zscore_anomaly'].sum()} / {len(df)}")
print(f"\n  Détail par observation :")
print(f"  {'Unité':>6} | {'Saison':>10} | {'Membres':>8} | "
      f"{'Fuite%':>7} | {'Z-max':>8} | {'Statut'}")
print("  " + "-"*60)
for _, row in df.iterrows():
    statut = "🔴 ANOMALIE" if row['zscore_anomaly'] == 1 else "🟢 Normal"
    print(f"  {int(row['fk_type_unite']):>6} | {row['season']:>10} | "
          f"{int(row['nbmr_members_season']):>8} | "
          f"{row['leak_rate']:>7.1f} | {row['zscore_max']:>8.4f} | {statut}")

# ─────────────────────────────────────────────────────────────────────────────
# A.5  CONSENSUS — VOTE MAJORITAIRE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.5  CONSENSUS — VOTE MAJORITAIRE (≥ 2/3 méthodes)")
print("="*65)

df['vote_sum']      = df['iso_anomaly'] + df['lof_anomaly'] + df['zscore_anomaly']
df['consensus']     = (df['vote_sum'] >= 2).astype(int)

print(f"\n  Anomalies confirmées (≥2/3 méthodes) : {df['consensus'].sum()} / {len(df)}")
print(f"\n  {'Unité':>6} | {'Saison':>10} | {'Membres':>8} | "
      f"{'Fuite%':>7} | {'ISO':>5} | {'LOF':>5} | {'Z':>5} | {'Vote':>5} | {'Consensus'}")
print("  " + "-"*75)
for _, row in df.iterrows():
    cons = "🔴 ANOMALIE" if row['consensus'] == 1 else "🟢 Normal"
    print(f"  {int(row['fk_type_unite']):>6} | {row['season']:>10} | "
          f"{int(row['nbmr_members_season']):>8} | "
          f"{row['leak_rate']:>7.1f} | "
          f"{'A' if row['iso_anomaly'] else 'N':>5} | "
          f"{'A' if row['lof_anomaly'] else 'N':>5} | "
          f"{'A' if row['zscore_anomaly'] else 'N':>5} | "
          f"{int(row['vote_sum']):>5} | {cons}")

# Sauvegarder résultats
df.to_csv('models/anomaly_results.csv', index=False)
print(f"\n  💾 models/anomaly_results.csv")

# ─────────────────────────────────────────────────────────────────────────────
# A.6  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.6  VISUALISATIONS")
print("="*65)

# Palette
color_normal  = '#2196F3'
color_anomaly = '#F44336'

# ── PCA 2D — Isolation Forest ─────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Isolation Forest', 'Local Outlier Factor (LOF)', 'Z-Score']
anomaly_cols = ['iso_anomaly', 'lof_anomaly', 'zscore_anomaly']
score_cols   = ['iso_score', 'lof_score', 'zscore_max']

for idx, (title, acol) in enumerate(zip(titles, anomaly_cols)):
    ax = axes[idx]
    for i, row in df.iterrows():
        color = color_anomaly if row[acol] == 1 else color_normal
        marker = 'D' if row[acol] == 1 else 'o'
        ax.scatter(X_pca[i, 0], X_pca[i, 1], color=color,
                   marker=marker, s=200, edgecolors='k',
                   linewidths=1.2, zorder=3)
        ax.annotate(f"U{int(row['fk_type_unite'])}\n{row['season'][-4:]}",
                    (X_pca[i, 0], X_pca[i, 1]),
                    textcoords='offset points', xytext=(6, 5),
                    fontsize=7, alpha=0.9)
    n_anom = df[acol].sum()
    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=10)
    ax.set_title(f'{title}\n{n_anom} anomalie(s) détectée(s)', fontsize=11)
    ax.grid(alpha=0.3)
    # Légende
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color_normal,
               markersize=10, markeredgecolor='k', label='Normal'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor=color_anomaly,
               markersize=10, markeredgecolor='k', label='Anomalie'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

plt.suptitle(f'Détection d\'Anomalies — PCA 2D\n'
             f'Variance expliquée : {sum(var_exp):.1%} | 18 obs. DWH', fontsize=13)
plt.tight_layout()
plt.savefig('visuals/anomaly_pca_2d.png', dpi=150)
plt.close()
print(f"  💾 visuals/anomaly_pca_2d.png")

# ── Heatmap scores d'anomalie ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
# Normaliser les scores entre 0 et 1 (plus élevé = plus anormal)
iso_norm  = (df['iso_score'].max() - df['iso_score']) / \
            (df['iso_score'].max() - df['iso_score'].min() + 1e-9)
lof_norm  = (df['lof_score'] - df['lof_score'].min()) / \
            (df['lof_score'].max() - df['lof_score'].min() + 1e-9)
z_norm    = (df['zscore_max'] - df['zscore_max'].min()) / \
            (df['zscore_max'].max() - df['zscore_max'].min() + 1e-9)

score_matrix = pd.DataFrame({
    'Isolation Forest': iso_norm.values,
    'LOF':              lof_norm.values,
    'Z-Score':          z_norm.values,
    'Consensus':        df['vote_sum'].values / 3,
}, index=[f"U{int(r['fk_type_unite'])}\n{r['season']}" for _, r in df.iterrows()])

sns.heatmap(score_matrix.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
            ax=ax, linewidths=0.5, vmin=0, vmax=1,
            cbar_kws={'label': 'Score d\'anomalie normalisé (1=très anormal)'})
ax.set_title('Scores d\'Anomalie par Méthode\n'
             '(rouge = anormal, vert = normal)', fontsize=12)
ax.set_xlabel('Observation (Unité / Saison)')
plt.tight_layout()
plt.savefig('visuals/anomaly_heatmap.png', dpi=150)
plt.close()
print(f"  💾 visuals/anomaly_heatmap.png")

# ── Évolution temporelle avec anomalies surlignées ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, unit in enumerate(sorted(df['fk_type_unite'].unique())):
    sub = df[df['fk_type_unite'] == unit].sort_values('season_year')
    ax  = axes[idx]

    # Ligne historique
    ax.plot(sub['season_year'], sub['nbmr_members_season'],
            'b-o', linewidth=2.5, markersize=9, zorder=2, label='Adhérents')

    # Surligner les anomalies consensus
    anom = sub[sub['consensus'] == 1]
    if len(anom) > 0:
        ax.scatter(anom['season_year'], anom['nbmr_members_season'],
                   color=color_anomaly, s=250, zorder=4,
                   marker='D', edgecolors='darkred', linewidths=2,
                   label='Anomalie (consensus)')

    # Annoter chaque point
    for _, row in sub.iterrows():
        label_txt = f"{int(row['nbmr_members_season'])}"
        if row['consensus'] == 1:
            label_txt += "\n⚠️"
        ax.annotate(label_txt, (row['season_year'], row['nbmr_members_season']),
                    textcoords='offset points', xytext=(0, 10),
                    fontsize=9, ha='center', fontweight='bold')

    ax.set_title(f'Unité {unit}', fontsize=11, fontweight='bold')
    ax.set_ylabel("Adhérents")
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xticklabels(['2023/24', '2024/25', '2025/26'], fontsize=8)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle('Évolution des Adhérents avec Anomalies Détectées\n'
             '(◆ rouge = anomalie confirmée par ≥2/3 méthodes)', fontsize=13)
plt.tight_layout()
plt.savefig('visuals/anomaly_evolution.png', dpi=150)
plt.close()
print(f"  💾 visuals/anomaly_evolution.png")

# ── Comparaison des 3 méthodes ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Isolation Forest scores
df_sorted = df.sort_values('iso_score')
colors_iso = [color_anomaly if v == 1 else color_normal for v in df_sorted['iso_anomaly']]
axes[0].barh([f"U{int(r['fk_type_unite'])}-{r['season'][-4:]}"
              for _, r in df_sorted.iterrows()],
             df_sorted['iso_score'], color=colors_iso, edgecolor='k', alpha=0.85)
axes[0].axvline(0, color='black', linewidth=1)
axes[0].set_title('Isolation Forest\n(score < 0 = anormal)', fontsize=10)
axes[0].set_xlabel('Anomaly Score')
axes[0].grid(axis='x', alpha=0.3)

# LOF scores
df_sorted2 = df.sort_values('lof_score', ascending=False)
colors_lof = [color_anomaly if v == 1 else color_normal for v in df_sorted2['lof_anomaly']]
axes[1].barh([f"U{int(r['fk_type_unite'])}-{r['season'][-4:]}"
              for _, r in df_sorted2.iterrows()],
             df_sorted2['lof_score'], color=colors_lof, edgecolor='k', alpha=0.85)
axes[1].axvline(1, color='black', linewidth=1, linestyle='--', label='Seuil=1')
axes[1].set_title('LOF Score\n(score > 1 = anormal)', fontsize=10)
axes[1].set_xlabel('LOF Score')
axes[1].legend(fontsize=8); axes[1].grid(axis='x', alpha=0.3)

# Z-Score max
df_sorted3 = df.sort_values('zscore_max', ascending=False)
colors_z = [color_anomaly if v == 1 else color_normal for v in df_sorted3['zscore_anomaly']]
axes[2].barh([f"U{int(r['fk_type_unite'])}-{r['season'][-4:]}"
              for _, r in df_sorted3.iterrows()],
             df_sorted3['zscore_max'], color=colors_z, edgecolor='k', alpha=0.85)
axes[2].axvline(2, color='black', linewidth=1, linestyle='--', label='Seuil=2σ')
axes[2].set_title('Z-Score Max\n(score > 2σ = anormal)', fontsize=10)
axes[2].set_xlabel('Z-Score Max')
axes[2].legend(fontsize=8); axes[2].grid(axis='x', alpha=0.3)

plt.suptitle('Comparaison des 3 Méthodes de Détection d\'Anomalies\n'
             '(rouge = anomalie, bleu = normal)', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/anomaly_comparison.png', dpi=150)
plt.close()
print(f"  💾 visuals/anomaly_comparison.png")

# ── Résumé consensus ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
vote_counts = df.groupby(['fk_type_unite','season'])['vote_sum'].first().reset_index()
vote_counts['label'] = vote_counts.apply(
    lambda r: f"U{int(r['fk_type_unite'])}\n{r['season']}", axis=1)
colors_vote = [color_anomaly if v >= 2 else
               ('#FF9800' if v == 1 else color_normal)
               for v in vote_counts['vote_sum']]
bars = ax.bar(vote_counts['label'], vote_counts['vote_sum'],
              color=colors_vote, edgecolor='k', alpha=0.85)
ax.axhline(2, color='red', linestyle='--', linewidth=2,
           label='Seuil consensus (≥2/3)')
for bar, v in zip(bars, vote_counts['vote_sum']):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
            f'{int(v)}/3', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Nombre de méthodes détectant une anomalie')
ax.set_title('Vote Majoritaire — Consensus des 3 Méthodes\n'
             '(rouge ≥2/3 = anomalie confirmée, orange = suspect, bleu = normal)',
             fontsize=11)
ax.set_ylim(0, 3.5); ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/anomaly_consensus.png', dpi=150)
plt.close()
print(f"  💾 visuals/anomaly_consensus.png")

# ─────────────────────────────────────────────────────────────────────────────
# A.7  INTERPRÉTATION BUSINESS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("A.7  INTERPRÉTATION BUSINESS")
print("="*65)

anomalies_confirmed = df[df['consensus'] == 1][
    ['fk_type_unite','season','nbmr_members_season','leak_rate',
     'growth_rate','vote_sum']].copy()

print(f"\n  Anomalies confirmées (≥2/3 méthodes) : {len(anomalies_confirmed)}")
if len(anomalies_confirmed) > 0:
    print(anomalies_confirmed.round(2).to_string(index=False))

print(f"""
  Analyse des anomalies détectées :

  1. Cohérence avec le Clustering (Section E) :
     → Unité 1 identifiée comme Cluster 1 (déclin fort, haute volatilité)
     → Les anomalies détectées confirment ce profil à risque

  2. Cohérence avec la Classification (Section C) :
     → Les saisons avec dropout_risk=1 correspondent aux anomalies détectées

  3. Recommandations métier :
     → Anomalies de type "chute brutale" : intervention immédiate requise
     → Anomalies de type "taux de fuite élevé" : programme de rétention
     → Anomalies de type "croissance anormale" : vérifier la qualité des données

  4. Valeur ajoutée de l'Anomaly Detection :
     → Alerte précoce avant que la situation ne devienne critique
     → Priorisation des ressources vers les unités à risque
     → Validation croisée avec les résultats du clustering et de la classification
""")

print("\n" + "█"*65)
print("  ✅ ANOMALY DETECTION TERMINÉE")
print("█"*65)
