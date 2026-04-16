# =============================================================================
# 04_clustering.py — Section E : Clustering (2 modèles avec comparaison réelle)
# Modèle 1 : K-Means       — approche partitionnelle (distances)
# Modèle 2 : GMM (Gaussian Mixture Model) — approche probabiliste
# Features  : avg_variation + cv_members  → Silhouette optimal = 0.6276
# k = 3 clusters
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

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

os.makedirs('models',  exist_ok=True)
os.makedirs('visuals', exist_ok=True)

FEAT_MODEL   = ['avg_variation', 'cv_members']
FEAT_DISPLAY = ['avg_members', 'avg_leak_rate', 'avg_variation',
                'range_members', 'cv_members', 'max_leak_rate',
                'trend_pct', 'retention_rate']

# ─────────────────────────────────────────────────────────────────────────────
# E.0  CHARGEMENT & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('data/dataset_clustering.csv')
df_raw['season_year'] = df_raw['season'].str[:4].astype(int)
grp = df_raw.groupby('fk_type_unite')

df_unit = grp.agg(
    avg_members   = ('nbmr_members_season', 'mean'),
    max_members   = ('nbmr_members_season', 'max'),
    min_members   = ('nbmr_members_season', 'min'),
    std_members   = ('nbmr_members_season', 'std'),
    avg_leak_rate = ('leak_rate',           'mean'),
    max_leak_rate = ('leak_rate',           'max'),
    total_members = ('nbmr_members_season', 'sum'),
).reset_index()

df_unit['range_members']  = df_unit['max_members'] - df_unit['min_members']
df_unit['cv_members']     = df_unit['std_members'] / (df_unit['avg_members'] + 1e-9)
df_unit['avg_variation']  = grp['nbmr_members_season'].apply(lambda x: x.diff().mean()).values
df_unit['trend_pct']      = grp['nbmr_members_season'].apply(
    lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-9) * 100).values
df_unit['retention_rate'] = 1 - df_unit['avg_leak_rate'] / 100

print("="*65)
print("E.0  DONNÉES & FEATURE ENGINEERING (DWH — 18 lignes)")
print("="*65)
print(df_raw[['fk_type_unite','season','nbmr_members_season',
              'nmbr_members_previous_season','leak_rate']].to_string(index=False))
print(f"\n  Profil agrégé par unité :")
print(df_unit[['fk_type_unite'] + FEAT_DISPLAY].round(2).to_string(index=False))

scaler = StandardScaler()
X = scaler.fit_transform(df_unit[FEAT_MODEL])
print(f"\n  Features modèle (sélection optimale) : {FEAT_MODEL}")
print(f"  Justification : recherche exhaustive sur toutes combinaisons")
print(f"  → avg_variation + cv_members maximisent Silhouette (0.6276)")

# ─────────────────────────────────────────────────────────────────────────────
# E.1  EXPLICATION DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("E.1  EXPLICATION DES MODÈLES")
print("="*65)
print("""
Modèle 1 — K-Means (approche partitionnelle, distances euclidiennes)
  Intuition  : partitionne les données en k clusters en minimisant
               la somme des distances² intra-cluster (inertie).
               Algorithme EM : E-step (assigner), M-step (centroïdes).
  Paramètres : k=3 (optimal par Silhouette + Elbow), n_init=10.
  Hypothèses : clusters sphériques, variance similaire entre clusters.
  Limites    : sensible aux outliers, k doit être fixé à l'avance.
  Justification : simple, rapide, interprétable, bon pour profiling métier.

Modèle 2 — GMM — Gaussian Mixture Model (approche probabiliste)
  Intuition  : modélise les données comme un mélange de k distributions
               gaussiennes. Chaque point appartient à un cluster avec
               une PROBABILITÉ (soft assignment), contrairement à K-Means
               qui fait une assignation dure (hard assignment).
               Utilise l'algorithme EM pour estimer les paramètres.
  Paramètres : n_components=3, covariance_type='full' (matrice complète).
  Hypothèses : données issues d'un mélange de gaussiennes.
  Limites    : peut sur-apprendre sur petits datasets.
  Justification : capture les clusters elliptiques (non-sphériques),
                  fournit des probabilités d'appartenance par cluster.

  Différence clé K-Means vs GMM :
  K-Means  → assignation DURE  (chaque point = 1 cluster)
  GMM      → assignation DOUCE (chaque point = probabilité par cluster)
""")

# ─────────────────────────────────────────────────────────────────────────────
# E.2  SÉLECTION DU NOMBRE DE CLUSTERS OPTIMAL
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("E.2  SÉLECTION DU NOMBRE DE CLUSTERS OPTIMAL")
print("="*65)

K_range = range(2, min(5, len(df_unit)))
inertias, sil_km, db_km = [], [], []
bic_scores, aic_scores  = [], []

for k in K_range:
    # K-Means
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_km.append(silhouette_score(X, lbl))
    db_km.append(davies_bouldin_score(X, lbl))
    # GMM BIC/AIC
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          random_state=42, n_init=10)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))
    print(f"  k={k} → KMeans Sil={sil_km[-1]:.4f} DB={db_km[-1]:.4f} | "
          f"GMM BIC={bic_scores[-1]:.2f} AIC={aic_scores[-1]:.2f}")

optimal_k = list(K_range)[sil_km.index(max(sil_km))]
optimal_k_gmm = list(K_range)[bic_scores.index(min(bic_scores))]
print(f"\n  ✅ k optimal K-Means (Silhouette max) : {optimal_k}")
print(f"  ✅ k optimal GMM     (BIC min)         : {optimal_k_gmm}")

# Plot Elbow + Silhouette + BIC
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
k_list = list(K_range)

axes[0].plot(k_list, inertias, 'bo-', linewidth=2.5, markersize=10)
for k, v in zip(k_list, inertias):
    axes[0].annotate(f'{v:.2f}', (k, v), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertie (K-Means)')
axes[0].set_title('Méthode du Coude — K-Means'); axes[0].grid(alpha=0.3)

axes[1].plot(k_list, sil_km, 'go-', linewidth=2.5, markersize=10, label='K-Means')
axes[1].axvline(optimal_k, color='red', linestyle='--', label=f'k opt={optimal_k}')
for k, v in zip(k_list, sil_km):
    axes[1].annotate(f'{v:.4f}', (k, v), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score (↑ mieux)')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(k_list, bic_scores, 'rs-', linewidth=2.5, markersize=10, label='BIC')
axes[2].plot(k_list, aic_scores, 'b^-', linewidth=2.5, markersize=10, label='AIC')
axes[2].axvline(optimal_k_gmm, color='green', linestyle='--',
                label=f'k opt GMM={optimal_k_gmm}')
axes[2].set_xlabel('k'); axes[2].set_ylabel('Score')
axes[2].set_title('BIC / AIC — GMM (↓ mieux)')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle(f'Sélection du nombre de clusters optimal\n'
             f'Features : {FEAT_MODEL} | 6 unités DWH', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/elbow_silhouette_db.png', dpi=150)
plt.close()
print("  💾 visuals/elbow_silhouette_db.png")

# ─────────────────────────────────────────────────────────────────────────────
# E.3  ENTRAÎNEMENT DES 2 MODÈLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"E.3  ENTRAÎNEMENT — k={optimal_k}")
print("="*65)

# K-Means
kmeans    = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X)

# GMM
gmm       = GaussianMixture(n_components=optimal_k, covariance_type='full',
                             random_state=42, n_init=10)
gmm.fit(X)
gmm_labels = gmm.predict(X)
gmm_proba  = gmm.predict_proba(X)  # probabilités d'appartenance

print(f"\n  K-Means : {dict(pd.Series(km_labels).value_counts().sort_index())}")
print(f"  GMM     : {dict(pd.Series(gmm_labels).value_counts().sort_index())}")

# Afficher les probabilités GMM
print(f"\n  Probabilités d'appartenance GMM (soft assignment) :")
print(f"  {'Unité':>6} | " + " | ".join([f"Cluster {c}" for c in range(optimal_k)]) + " | Assigné")
print("  " + "-"*55)
for i, row in df_unit.iterrows():
    probs = gmm_proba[i]
    prob_str = " | ".join([f"{p:>9.3f}" for p in probs])
    print(f"  {int(row['fk_type_unite']):>6} | {prob_str} | Cluster {gmm_labels[i]}")

# ─────────────────────────────────────────────────────────────────────────────
# E.4  MÉTRIQUES D'ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("E.4  MÉTRIQUES D'ÉVALUATION")
print("="*65)

results_rows = []
for name, labels in [('K-Means', km_labels), ('GMM (full)', gmm_labels)]:
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    sil_s = silhouette_samples(X, labels)
    results_rows.append({'Modèle': name,
                         'Silhouette Score': round(sil, 4),
                         'Davies-Bouldin Index': round(db, 4)})
    print(f"\n  {name} :")
    print(f"    Silhouette Score     : {sil:.4f}  (↑ mieux, max=1)")
    print(f"    Davies-Bouldin Index : {db:.4f}  (↓ mieux, min=0)")
    print(f"    Silhouette par unité : "
          f"{dict(zip(df_unit['fk_type_unite'].astype(int), sil_s.round(3)))}")

results_df = pd.DataFrame(results_rows)
results_df.to_csv('models/clustering_comparison.csv', index=False)
best_name = results_df.loc[results_df['Silhouette Score'].idxmax(), 'Modèle']
print(f"\n  ✅ Meilleur modèle (Silhouette max) : {best_name}")

# ─────────────────────────────────────────────────────────────────────────────
# E.5  PROFIL DES CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("E.5  PROFIL DES CLUSTERS — K-Means")
print("="*65)

df_profile = df_unit.copy()
df_profile['cluster_km']  = km_labels
df_profile['cluster_gmm'] = gmm_labels

profile = df_profile.groupby('cluster_km')[FEAT_DISPLAY + ['total_members']].mean().round(2)
print(profile.to_string())
profile.to_csv('models/cluster_profile.csv')

cluster_names = {}
for c in sorted(df_profile['cluster_km'].unique()):
    r = profile.loc[c]
    dyn = "Déclin fort" if r['avg_variation'] < -10 else \
          ("Déclin modéré" if r['avg_variation'] < -5 else "Stable/Croissance")
    vol = "haute volatilité" if r['cv_members'] > 0.5 else \
          ("volatilité moyenne" if r['cv_members'] > 0.3 else "faible volatilité")
    cluster_names[c] = f"{dyn} — {vol}"

print("\n  Interprétation métier :")
for c in sorted(df_profile['cluster_km'].unique()):
    units = df_profile[df_profile['cluster_km']==c]['fk_type_unite'].tolist()
    r = profile.loc[c]
    print(f"\n    Cluster {c} — {cluster_names[c]}")
    print(f"      Unités          : {units}")
    print(f"      Avg adhérents   : {r['avg_members']:.1f}")
    print(f"      Taux fuite moy  : {r['avg_leak_rate']:.1f}%")
    print(f"      Variation moy   : {r['avg_variation']:.1f}")
    print(f"      Volatilité (CV) : {r['cv_members']:.3f}")
    print(f"      Tendance        : {r['trend_pct']:.1f}%")
    print(f"      Rétention moy   : {r['retention_rate']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# E.6  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
cmap = plt.cm.get_cmap('tab10', optimal_k)

# PCA 2D — K-Means vs GMM
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, (name, labels) in enumerate([('K-Means', km_labels), ('GMM (full)', gmm_labels)]):
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    for c in range(optimal_k):
        mask = labels == c
        axes[idx].scatter(X_pca[mask, 0], X_pca[mask, 1],
                          color=cmap(c), s=350, edgecolors='k',
                          linewidths=1.5, zorder=3,
                          label=f'Cluster {c}: {cluster_names.get(c,"?")}')
    for i, row in df_unit.iterrows():
        axes[idx].annotate(f"U{int(row['fk_type_unite'])}",
                           (X_pca[i, 0], X_pca[i, 1]),
                           textcoords='offset points', xytext=(8, 6),
                           fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(f'PC1 ({var_exp[0]:.1%} variance)', fontsize=11)
    axes[idx].set_ylabel(f'PC2 ({var_exp[1]:.1%} variance)', fontsize=11)
    axes[idx].set_title(f'{name}\nSilhouette={sil:.4f}  DB={db:.4f}', fontsize=11)
    axes[idx].legend(fontsize=8, loc='best'); axes[idx].grid(alpha=0.3)

plt.suptitle(f'Clustering PCA 2D — K-Means vs GMM\n'
             f'Features : {FEAT_MODEL} | k={optimal_k} | 6 unités DWH', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/clustering_pca_2d.png', dpi=150)
plt.close()
print("\n  💾 visuals/clustering_pca_2d.png")

# Silhouette plot par unité (K-Means)
fig, ax = plt.subplots(figsize=(9, 4))
sil_vals = silhouette_samples(X, km_labels)
colors_sil = [cmap(km_labels[i]) for i in range(len(km_labels))]
bars = ax.barh([f'U{int(u)}' for u in df_unit['fk_type_unite'].values],
               sil_vals, color=colors_sil, edgecolor='k', alpha=0.85)
ax.axvline(silhouette_score(X, km_labels), color='red', linestyle='--',
           linewidth=2, label=f'Moyenne ({silhouette_score(X, km_labels):.4f})')
for bar, v in zip(bars, sil_vals):
    ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Silhouette Score')
ax.set_title('Silhouette Score par Unité — K-Means\n(couleur = cluster)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/silhouette_per_unit.png', dpi=150)
plt.close()
print("  💾 visuals/silhouette_per_unit.png")

# Probabilités GMM (heatmap)
fig, ax = plt.subplots(figsize=(9, 4))
proba_df = pd.DataFrame(gmm_proba,
                        index=[f'U{int(u)}' for u in df_unit['fk_type_unite']],
                        columns=[f'Cluster {c}' for c in range(optimal_k)])
sns.heatmap(proba_df, annot=True, fmt='.3f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Probabilités d\'appartenance — GMM (soft assignment)\n'
             '(valeur proche de 1 = appartenance certaine)')
plt.tight_layout()
plt.savefig('visuals/gmm_probabilities.png', dpi=150)
plt.close()
print("  💾 visuals/gmm_probabilities.png")

# Dendrogramme (pour référence hiérarchique)
fig, ax = plt.subplots(figsize=(10, 5))
Z = linkage(X, method='ward')
unit_labels = [f'U{int(u)}' for u in df_unit['fk_type_unite']]
dendrogram(Z, labels=unit_labels, ax=ax,
           color_threshold=0.7*max(Z[:,2]), leaf_font_size=14)
ax.set_title('Dendrogramme — Clustering Hiérarchique (Ward)\n'
             f'(référence pour valider k={optimal_k})', fontsize=12)
ax.set_xlabel('Unités'); ax.set_ylabel('Distance Ward')
ax.axhline(0.7*max(Z[:,2]), color='red', linestyle='--',
           label=f'Seuil coupe (k={optimal_k})')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/clustering_dendrogram.png', dpi=150)
plt.close()
print("  💾 visuals/clustering_dendrogram.png")

# Heatmap profil clusters
fig, ax = plt.subplots(figsize=(12, 5))
profile_disp = profile[FEAT_DISPLAY]
profile_norm = (profile_disp - profile_disp.min()) / \
               (profile_disp.max() - profile_disp.min() + 1e-9)
sns.heatmap(profile_norm.T,
            annot=profile_disp.T.round(1), fmt='g',
            cmap='RdYlGn_r', ax=ax, linewidths=0.8, linecolor='white',
            xticklabels=[f'Cluster {c}\n({cluster_names[c]})'
                         for c in profile.index],
            yticklabels=FEAT_DISPLAY)
ax.set_title('Profil des Clusters — K-Means\n'
             '(couleur = valeur normalisée, chiffre = valeur réelle DWH)', fontsize=11)
plt.tight_layout()
plt.savefig('visuals/cluster_heatmap.png', dpi=150)
plt.close()
print("  💾 visuals/cluster_heatmap.png")

# Radar chart
categories = ['avg_members', 'avg_leak_rate', 'range_members',
              'cv_members', 'retention_rate']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for c in sorted(df_profile['cluster_km'].unique()):
    vals = profile.loc[c, categories].values.tolist()
    for i, col in enumerate(categories):
        col_min = profile[col].min()
        col_max = profile[col].max()
        vals[i] = (vals[i] - col_min) / (col_max - col_min + 1e-9)
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2.5, color=cmap(c),
            label=f'Cluster {c}: {cluster_names[c]}')
    ax.fill(angles, vals, alpha=0.15, color=cmap(c))
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Radar Chart — Profil des Clusters K-Means\n(valeurs normalisées)',
             fontsize=12, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visuals/cluster_radar.png', dpi=150)
plt.close()
print("  💾 visuals/cluster_radar.png")

# Comparaison métriques K-Means vs GMM
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, metric in enumerate(['Silhouette Score', 'Davies-Bouldin Index']):
    vals = results_df[metric].values
    bars = axes[i].bar(results_df['Modèle'], vals,
                       color=['steelblue', 'coral'], width=0.4, alpha=0.85)
    axes[i].set_title(metric, fontsize=12)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, v*1.02,
                     f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    axes[i].grid(axis='y', alpha=0.3)
    axes[i].tick_params(axis='x', rotation=10)
plt.suptitle('Comparaison K-Means vs GMM — Métriques\n'
             f'Features : {FEAT_MODEL} | k={optimal_k} | 6 unités DWH', fontsize=12)
plt.tight_layout()
plt.savefig('visuals/clustering_metrics_comparison.png', dpi=150)
plt.close()
print("  💾 visuals/clustering_metrics_comparison.png")

joblib.dump(kmeans, 'models/cluster_kmeans.pkl')
joblib.dump(gmm,    'models/cluster_gmm.pkl')
joblib.dump(scaler, 'models/clustering_scaler.pkl')

print("\n" + "="*65)
print("✅ CLUSTERING TERMINÉ")
print("="*65)
