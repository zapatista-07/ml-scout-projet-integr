# =============================================================================
# generate_data.py — Génération de données synthétiques réalistes
# Étend les datasets de 18 → ~120 lignes en respectant les patterns réels
# =============================================================================
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres réels observés par unité (base = données réelles)
# ─────────────────────────────────────────────────────────────────────────────
# Profil de chaque unité : (membres_base, tendance_annuelle, volatilité)
unit_profiles = {
    1: {'base': 45, 'trend': -0.35, 'vol': 0.12, 'leak_base': 32, 'leak_vol': 10},
    2: {'base': 40, 'trend': -0.20, 'vol': 0.15, 'leak_base': 18, 'leak_vol': 15},
    3: {'base': 24, 'trend': -0.18, 'vol': 0.10, 'leak_base': 19, 'leak_vol': 8},
    4: {'base': 34, 'trend': -0.15, 'vol': 0.12, 'leak_base': 16, 'leak_vol': 10},
    5: {'base': 12, 'trend': -0.10, 'vol': 0.08, 'leak_base': 12, 'leak_vol': 6},
    6: {'base':  8, 'trend': -0.08, 'vol': 0.10, 'leak_base': 13, 'leak_vol': 8},
}

# Saisons réelles + saisons synthétiques étendues
real_seasons = [
    (2023, '2023/2024'),
    (2024, '2024/2025'),
    (2025, '2025/2026'),
]
# On génère 7 saisons supplémentaires en arrière (2016-2022) pour avoir de l'historique
synthetic_seasons = [(y, f'{y}/{y+1}') for y in range(2016, 2023)]
all_seasons = synthetic_seasons + real_seasons  # 10 saisons total

rows_reg, rows_cls, rows_clu = [], [], []

for unit_id, profile in unit_profiles.items():
    prev_members = None

    for i, (year, season_str) in enumerate(all_seasons):
        # Calcul membres avec tendance + bruit
        t = i / (len(all_seasons) - 1)  # 0 → 1
        base_val = max(3, profile['base'] * (1 + profile['trend'] * i))
        noise = np.random.normal(0, profile['vol'] * base_val)
        members = max(3, int(round(base_val + noise)))

        # Saison précédente
        prev = 0 if prev_members is None else prev_members

        # Leak rate
        if prev > 0:
            lost = max(0, prev - members + np.random.randint(-2, 3))
            leak = min(95.0, round((lost / prev) * 100 + np.random.normal(0, 3), 2))
            leak = max(0.0, leak)
        else:
            leak = 0.0

        # dropout_risk : 1 si membres < précédents ET variation > -10%
        if prev > 0:
            variation = (members - prev) / prev
            dropout = 1 if variation < -0.05 else 0
        else:
            dropout = 0

        # REGRESSION
        rows_reg.append({
            'fk_type_unite': unit_id,
            'nmbr_members_previous_season': prev,
            'season_year': year,
            'nbmr_members_season': members,
        })

        # CLASSIFICATION
        rows_cls.append({
            'fk_type_unite': unit_id,
            'nbmr_members_season': members,
            'nmbr_members_previous_season': prev,
            'season_year': year,
            'dropout_risk': dropout,
        })

        # CLUSTERING
        rows_clu.append({
            'nbmr_members_season': members,
            'nmbr_members_previous_season': prev,
            'leak_rate': leak,
            'fk_type_unite': unit_id,
            'season': season_str,
        })

        prev_members = members

# ─────────────────────────────────────────────────────────────────────────────
# Créer DataFrames
# ─────────────────────────────────────────────────────────────────────────────
df_reg = pd.DataFrame(rows_reg)
df_cls = pd.DataFrame(rows_cls)
df_clu = pd.DataFrame(rows_clu)

# Remplacer les 3 dernières saisons par les vraies valeurs (cohérence)
real_data = {
    (1,2023):45,(1,2024):25,(1,2025):12,
    (2,2023):40,(2,2024):18,(2,2025):25,
    (3,2023):24,(3,2024):16,(3,2025):12,
    (4,2023):34,(4,2024):18,(4,2025):20,
    (5,2023):12,(5,2024):10,(5,2025):8,
    (6,2023):8, (6,2024):10,(6,2025):6,
}
for (unit, year), val in real_data.items():
    mask = (df_reg['fk_type_unite']==unit) & (df_reg['season_year']==year)
    df_reg.loc[mask, 'nbmr_members_season'] = val
    df_cls.loc[mask, 'nbmr_members_season'] = val
    df_clu.loc[mask, 'nbmr_members_season'] = val

# Recalculer dropout_risk sur les vraies valeurs
for idx, row in df_cls.iterrows():
    prev = row['nmbr_members_previous_season']
    curr = row['nbmr_members_season']
    if prev > 0:
        df_cls.loc[idx, 'dropout_risk'] = 1 if (curr - prev) / prev < -0.05 else 0
    else:
        df_cls.loc[idx, 'dropout_risk'] = 0

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarder
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)

# Garder les originaux en backup
import shutil
for f in ['dataset_regression.csv','dataset_classification.csv','dataset_clustering.csv']:
    if os.path.exists(f'data/{f}') and not os.path.exists(f'data/original_{f}'):
        shutil.copy(f'data/{f}', f'data/original_{f}')

df_reg.to_csv('data/dataset_regression.csv', index=False)
df_cls.to_csv('data/dataset_classification.csv', index=False)
df_clu.to_csv('data/dataset_clustering.csv', index=False)

print("="*55)
print("DATASETS GÉNÉRÉS")
print("="*55)
print(f"Régression    : {df_reg.shape}  | Missing: {df_reg.isnull().sum().sum()}")
print(f"Classification: {df_cls.shape}  | Missing: {df_cls.isnull().sum().sum()}")
print(f"Clustering    : {df_clu.shape}  | Missing: {df_clu.isnull().sum().sum()}")

print(f"\nDistribution dropout_risk :\n{df_cls['dropout_risk'].value_counts().to_string()}")
print(f"\nRégression — stats nbmr_members_season :\n{df_reg['nbmr_members_season'].describe().round(2).to_string()}")
print(f"\nSaisons couvertes : {sorted(df_reg['season_year'].unique())}")
print("\n✅ Backups originaux sauvegardés dans data/original_*.csv")
