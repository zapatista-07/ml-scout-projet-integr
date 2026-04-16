import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from itertools import combinations

df_raw = pd.read_csv('data/dataset_clustering.csv')
df_raw['season_year'] = df_raw['season'].str[:4].astype(int)
grp = df_raw.groupby('fk_type_unite')

df_unit = grp.agg(
    avg_members   = ('nbmr_members_season', 'mean'),
    max_members   = ('nbmr_members_season', 'max'),
    min_members   = ('nbmr_members_season', 'min'),
    std_members   = ('nbmr_members_season', 'std'),
    avg_leak      = ('leak_rate',           'mean'),
    max_leak      = ('leak_rate',           'max'),
    total         = ('nbmr_members_season', 'sum'),
).reset_index()
df_unit['range_members']  = df_unit['max_members'] - df_unit['min_members']
df_unit['cv_members']     = df_unit['std_members'] / (df_unit['avg_members'] + 1e-9)
df_unit['avg_variation']  = grp['nbmr_members_season'].apply(lambda x: x.diff().mean()).values
df_unit['trend_pct']      = grp['nbmr_members_season'].apply(
    lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-9) * 100).values
df_unit['retention_rate'] = 1 - df_unit['avg_leak'] / 100

feat_all = ['avg_members', 'avg_leak', 'avg_variation', 'range_members',
            'cv_members', 'max_leak', 'trend_pct', 'retention_rate']
scaler = StandardScaler()

results = []
for k in [2, 3]:
    for n_feat in range(2, 7):
        for feat_combo in combinations(feat_all, n_feat):
            X = scaler.fit_transform(df_unit[list(feat_combo)])
            for model_name, model in [('KMeans', KMeans(n_clusters=k, random_state=42, n_init=10)),
                                       ('HC_Ward', AgglomerativeClustering(n_clusters=k, linkage='ward'))]:
                try:
                    lbl = model.fit_predict(X)
                    if len(set(lbl)) < k:
                        continue
                    sil = silhouette_score(X, lbl)
                    db  = davies_bouldin_score(X, lbl)
                    results.append({
                        'model': model_name, 'k': k,
                        'features': list(feat_combo),
                        'sil': round(sil, 4), 'db': round(db, 4),
                        'labels': lbl.tolist()
                    })
                except:
                    pass

results.sort(key=lambda x: x['sil'], reverse=True)
print('Top 15 configurations (Silhouette max):')
for i, r in enumerate(results[:15]):
    print(f"\n  #{i+1} {r['model']} k={r['k']} | Sil={r['sil']} | DB={r['db']}")
    print(f"       Features: {r['features']}")
    print(f"       Labels  : {r['labels']}")
