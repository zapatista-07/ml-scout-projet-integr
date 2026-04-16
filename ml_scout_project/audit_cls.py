import pandas as pd
import numpy as np

df_cls = pd.read_csv('data/dataset_classification.csv')
df_clu = pd.read_csv('data/dataset_clustering.csv')
df_clu['season_year'] = df_clu['season'].str[:4].astype(int)

df = df_cls.merge(df_clu[['fk_type_unite','season_year','leak_rate']],
                  on=['fk_type_unite','season_year'], how='left')
df = df.sort_values(['fk_type_unite','season_year']).reset_index(drop=True)

grp = df.groupby('fk_type_unite')
df['members_lag1']           = df['nmbr_members_previous_season']
df['leak_rate_lag1']         = grp['leak_rate'].shift(1).fillna(0)
df['growth_rate_lag1']       = grp['nbmr_members_season'].pct_change().shift(1).fillna(0)*100
df['delta_members_lag1']     = grp['nbmr_members_season'].diff().shift(1).fillna(0)
df['avg_members_unite_past'] = grp['nbmr_members_season'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.iloc[0]))
df['season_index']           = grp.cumcount() + 1
df['retention_rate']         = 1 - df['leak_rate_lag1']/100
df['members_x_retention']    = df['members_lag1'] * df['retention_rate']

cols = ['fk_type_unite','season_year','members_lag1','leak_rate_lag1',
        'growth_rate_lag1','delta_members_lag1','avg_members_unite_past',
        'season_index','retention_rate','members_x_retention','dropout_risk']
print('Dataset 18 lignes avec dropout_risk:')
print(df[cols].round(2).to_string(index=False))

print('\nDistribution dropout_risk:')
print(df['dropout_risk'].value_counts().to_string())

feat = ['members_lag1','leak_rate_lag1','growth_rate_lag1','delta_members_lag1',
        'avg_members_unite_past','season_index','fk_type_unite',
        'retention_rate','members_x_retention']
print('\nCorrelations avec dropout_risk:')
corr = df[feat+['dropout_risk']].corr()['dropout_risk'].drop('dropout_risk').sort_values(ascending=False)
print(corr.round(4).to_string())
