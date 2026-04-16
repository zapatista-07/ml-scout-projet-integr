import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

df_raw = pd.read_csv('data/dataset_clustering.csv')
df_raw['season_year'] = df_raw['season'].str[:4].astype(int)
grp = df_raw.groupby('fk_type_unite')

df_unit = grp.agg(
    avg_members   = ('nbmr_members_season', 'mean'),
    std_members   = ('nbmr_members_season', 'std'),
    avg_leak_rate = ('leak_rate',           'mean'),
    max_leak_rate = ('leak_rate',           'max'),
).reset_index()
df_unit['cv_members']    = df_unit['std_members'] / (df_unit['avg_members'] + 1e-9)
df_unit['avg_variation'] = grp['nbmr_members_season'].apply(lambda x: x.diff().mean()).values
df_unit['trend_pct']     = grp['nbmr_members_season'].apply(
    lambda x: (x.iloc[-1]-x.iloc[0])/(x.iloc[0]+1e-9)*100).values
df_unit['retention_rate']= 1 - df_unit['avg_leak_rate']/100

FEAT = ['avg_variation', 'cv_members']
scaler = StandardScaler()
X = scaler.fit_transform(df_unit[FEAT])

print("Comparaison K-Means vs GMM sur k=2,3,4:")
print(f"{'Modèle':20s} {'k':>3} {'Silhouette':>12} {'DB':>10} {'Labels'}")
print("-"*70)

for k in [2, 3]:
    # K-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl_km = km.fit_predict(X)
    sil_km = silhouette_score(X, lbl_km)
    db_km  = davies_bouldin_score(X, lbl_km)
    print(f"{'K-Means':20s} {k:>3} {sil_km:>12.4f} {db_km:>10.4f}  {lbl_km.tolist()}")

    # GMM avec differents covariance types
    for cov in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=42, n_init=10)
            gmm.fit(X)
            lbl_gmm = gmm.predict(X)
            if len(set(lbl_gmm)) < k:
                continue
            sil_gmm = silhouette_score(X, lbl_gmm)
            db_gmm  = davies_bouldin_score(X, lbl_gmm)
            diff = "DIFFERENT" if not np.array_equal(
                np.sort(lbl_km), np.sort(lbl_gmm)) else "same"
            print(f"{'GMM-'+cov:20s} {k:>3} {sil_gmm:>12.4f} {db_gmm:>10.4f}  "
                  f"{lbl_gmm.tolist()}  [{diff}]")
        except Exception as e:
            print(f"  GMM-{cov} k={k} failed: {e}")
    print()
