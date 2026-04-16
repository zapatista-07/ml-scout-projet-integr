# =============================================================================
# build_features.py — Construction du dataset avec features laggées
# "Pas de nouvelles données — uniquement des indicateurs dérivés du DWH"
#
# Logique :
#   Pour prédire la saison N, on utilise les valeurs de la saison N-1 (lag1)
#   Cela permet de prédire 2026/2027 à partir des données 2025/2026 connues.
#
# Features construites :
#   members_lag1        → nbmr_members_season de la saison précédente
#   leak_rate_lag1      → leak_rate de la saison précédente
#   growth_rate_lag1    → taux de croissance de la saison précédente
#   delta_members_lag1  → delta absolu membres de la saison précédente
#   avg_members_unite_past → moyenne historique des membres de l'unité
#   season_index        → numéro de saison (1=2023/24, 2=2024/25, 3=2025/26)
#   fk_type_unite       → identifiant de l'unité
# =============================================================================
import pandas as pd
import numpy as np
import os

# ── Charger les deux sources DWH ─────────────────────────────────────────────
df_reg = pd.read_csv('data/dataset_regression.csv')
df_clu = pd.read_csv('data/dataset_clustering.csv')

# Fusionner pour avoir leak_rate
df = df_reg.merge(
    df_clu[['fk_type_unite','season','leak_rate']],
    left_on=['fk_type_unite','season_year'],
    right_on=['fk_type_unite', df_clu['season'].str[:4].astype(int)],
    how='left'
)

# Nettoyage : reconstruire proprement
df_base = df_clu.copy()
df_base['season_year'] = df_base['season'].str[:4].astype(int)
df_base = df_base.merge(
    df_reg[['fk_type_unite','season_year','nbmr_members_season']],
    on=['fk_type_unite','season_year'], how='left'
)
# Utiliser nbmr_members_season du clustering (identique)
df_base['nbmr_members_season'] = df_base['nbmr_members_season_x'].fillna(
    df_base['nbmr_members_season_y'])
df_base = df_base[['fk_type_unite','season','season_year',
                   'nbmr_members_season','leak_rate']].copy()
df_base = df_base.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

print("Dataset de base (DWH) :")
print(df_base.to_string(index=False))

# ── Construction des features laggées ────────────────────────────────────────
grp = df_base.groupby('fk_type_unite')

# 1. members_lag1 : membres de la saison précédente
df_base['members_lag1'] = grp['nbmr_members_season'].shift(1)

# 2. leak_rate_lag1 : taux de fuite de la saison précédente
df_base['leak_rate_lag1'] = grp['leak_rate'].shift(1)

# 3. growth_rate_lag1 : taux de croissance saison précédente
#    = (membres_t-1 - membres_t-2) / membres_t-2
df_base['growth_rate_lag1'] = grp['nbmr_members_season'].pct_change().shift(1) * 100

# 4. delta_members_lag1 : variation absolue saison précédente
#    = membres_t-1 - membres_t-2
df_base['delta_members_lag1'] = grp['nbmr_members_season'].diff().shift(1)

# 5. avg_members_unite_past : moyenne historique de l'unité
#    = moyenne de toutes les saisons PASSÉES (expanding, sans fuite du futur)
df_base['avg_members_unite_past'] = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# 6. season_index : rang de la saison (1, 2, 3...)
df_base['season_index'] = grp.cumcount() + 1

# 7. fk_type_unite : déjà présent

# ── Supprimer les lignes sans lag (première saison de chaque unité) ──────────
df_feat = df_base.dropna(subset=['members_lag1']).reset_index(drop=True)

print(f"\nDataset avec features laggées ({df_feat.shape[0]} lignes) :")
cols_show = ['fk_type_unite','season','season_year','nbmr_members_season',
             'members_lag1','leak_rate_lag1','growth_rate_lag1',
             'delta_members_lag1','avg_members_unite_past','season_index']
print(df_feat[cols_show].round(2).to_string(index=False))

# ── Construire la ligne de prédiction 2026/2027 ──────────────────────────────
# Pour prédire 2026/2027, on utilise les valeurs 2025/2026 comme lag1
last_season = df_base[df_base['season_year'] == 2025].copy()

pred_rows = []
for _, row in last_season.iterrows():
    unit = row['fk_type_unite']
    hist = df_base[df_base['fk_type_unite'] == unit].sort_values('season_year')

    # growth_rate de 2025/2026 = (membres_2025 - membres_2024) / membres_2024
    if len(hist) >= 2:
        m_curr = hist.iloc[-1]['nbmr_members_season']
        m_prev = hist.iloc[-2]['nbmr_members_season']
        growth = (m_curr - m_prev) / m_prev * 100 if m_prev > 0 else 0.0
        delta  = m_curr - m_prev
    else:
        growth = 0.0
        delta  = 0.0

    pred_rows.append({
        'fk_type_unite':          unit,
        'season':                 '2026/2027',
        'season_year':            2026,
        'nbmr_members_season':    np.nan,          # cible à prédire
        'members_lag1':           row['nbmr_members_season'],
        'leak_rate_lag1':         row['leak_rate'],
        'growth_rate_lag1':       growth,
        'delta_members_lag1':     delta,
        'avg_members_unite_past': hist['nbmr_members_season'].mean(),
        'season_index':           4,               # 4ème saison
    })

df_pred = pd.DataFrame(pred_rows)
print(f"\nLignes de prédiction 2026/2027 :")
print(df_pred[cols_show].round(2).to_string(index=False))

# ── Sauvegarder ──────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
df_feat.to_csv('data/dataset_features.csv', index=False)
df_pred.to_csv('data/dataset_predict_2026.csv', index=False)

print(f"\n✅ data/dataset_features.csv  ({df_feat.shape[0]} lignes d'entraînement)")
print(f"✅ data/dataset_predict_2026.csv  ({df_pred.shape[0]} lignes à prédire)")
