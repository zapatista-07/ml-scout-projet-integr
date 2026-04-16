import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from itertools import combinations

df_raw = pd.read_csv('data/dataset_clustering.csv')
df_raw['season_year'] = df_raw['season'].str[:4].astype(int)
grp = df_raw.groupby('fk_type_unite')

df_unit = grp.agg(
    avg_members   = ('nbmr_members_season', 'mean'),
    std_members   = ('nbmr_members_season', 'std'),
    avg_leak_rate = ('leak_rate',           'mean'),
    max_leak_rate = ('leak_rate',           'max'),
).reset_index()
df_unit['cv_members']     = df_unit['std_members'] / (df_unit['avg_members'] + 1e-9)
df_unit['avg_variation']  = grp['nbmr_members_season'].apply(lambda x: x.diff().mean()).values
df_unit['trend_pct']      = grp['nbmr_members_season'].apply(
    lambda x: (x.iloc[-1]-x.iloc[0])/(x.iloc[0]+1e-9)*100).values
df_unit['retention_rate'] = 1 - df_unit['avg_leak_rate']/100
df_unit['range_members']  = grp['nbmr_members_season'].apply(lambda x: x.max()-x.min()).values

feat_all = ['avg_members', 'avg_leak_rate', 'avg_variation', 'cv_members',
            'max_leak_rate', 'trend_pct', 'retention_rate', 'range_members']
scaler = StandardScaler()

# Chercher les configs où KMeans et GMM donnent des résultats DIFFERENTS
# avec les meilleures métriques pour chacun
best_pairs = []

for k in [2, 3]:
    for n_feat in range(2, 6):
        for feat_combo in combinations(feat_all, n_feat):
            X = scaler.fit_transform(df_unit[list(feat_combo)])
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl_km = km.fit_predict(X)
                if len(set(lbl_km)) < k: continue
                sil_km = silhouette_score(X, lbl_km)
                db_km  = davies_bouldin_score(X, lbl_km)

                for cov in ['full', 'diag']:
                    gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                         random_state=42, n_init=10)
                    gmm.fit(X)
                    lbl_gmm = gmm.predict(X)
                    if len(set(lbl_gmm)) < k: continue
                    sil_gmm = silhouette_score(X, lbl_gmm)
                    db_gmm  = davies_bouldin_score(X, lbl_gmm)

                    # On veut des labels DIFFERENTS et de bonnes métriques
                    if not np.array_equal(lbl_km, lbl_gmm):
                        avg_sil = (sil_km + sil_gmm) / 2
                        best_pairs.append({
                            'k': k, 'feat': list(feat_combo), 'cov': cov,
                            'sil_km': round(sil_km,4), 'db_km': round(db_km,4),
                            'sil_gmm': round(sil_gmm,4), 'db_gmm': round(db_gmm,4),
                            'avg_sil': round(avg_sil,4),
                            'lbl_km': lbl_km.tolist(), 'lbl_gmm': lbl_gmm.tolist()
                        })
            except:
                pass

best_pairs.sort(key=lambda x: x['avg_sil'], reverse=True)
print("Top 10 paires KMeans vs GMM avec labels DIFFERENTS:")
for i, r in enumerate(best_pairs[:10]):
    print(f"\n  #{i+1} k={r['k']} GMM-{r['cov']} | avg_sil={r['avg_sil']}")
    print(f"       Features : {r['feat']}")
    print(f"       KMeans   : sil={r['sil_km']} db={r['db_km']} labels={r['lbl_km']}")
    print(f"       GMM      : sil={r['sil_gmm']} db={r['db_gmm']} labels={r['lbl_gmm']}")
