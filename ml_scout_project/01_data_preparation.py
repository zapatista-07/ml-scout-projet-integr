# 01_data_preparation.py (CORRIGÉ - Sans Data Leakage)
# Section A: Data Preparation & Feature Engineering

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import get_connection_string
import os

def load_data():
    print("🔌 Connexion à DWH_scout...")
    engine = create_engine(get_connection_string())
    query = text("""
    SELECT 
        fk_type_unite,
        season,
        nbmr_members_season,
        nmbr_members_previous_season,
        retained_members_count,
        leak_rate
    FROM fact_registrations
    ORDER BY fk_type_unite, season
    """)
    df = pd.read_sql_query(query, engine)
    engine.dispose()
    return df

def data_cleaning(df):
    print("\n" + "="*60)
    print("A.1 DATA CLEANING")
    print("="*60)
    print(f"\n📊 Valeurs manquantes avant: {df.isnull().sum().sum()}")
    df = df.fillna(0)
    print(f"📊 Valeurs manquantes après: {df.isnull().sum().sum()}")

    Q1 = df['leak_rate'].quantile(0.25)
    Q3 = df['leak_rate'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df['leak_rate'] < (Q1 - 1.5 * IQR)) |
                (df['leak_rate'] > (Q3 + 1.5 * IQR))).sum()
    print(f"📊 Outliers détectés (leak_rate): {outliers}")

    df['fk_type_unite'] = df['fk_type_unite'].astype(int)
    df['nbmr_members_season'] = df['nbmr_members_season'].astype(int)
    return df

def feature_engineering(df):
    print("\n" + "="*60)
    print("A.2 FEATURE ENGINEERING")
    print("="*60)

    # Encodage saison → numérique
    df['season_year'] = df['season'].str.split('/').str[0].astype(int)

    # Target Classification : risque dropout (leak_rate > 30%)
    # ⚠️ leak_rate est utilisé UNIQUEMENT pour créer le target, pas comme feature
    df['dropout_risk'] = (df['leak_rate'] > 30).astype(int)

    print("✅ Features créées: season_year, dropout_risk")
    print(f"📊 Distribution dropout_risk:\n{df['dropout_risk'].value_counts()}")
    return df

def feature_selection(df):
    print("\n" + "="*60)
    print("A.3 FEATURE SELECTION (Sans Data Leakage)")
    print("="*60)

    # ✅ REGRESSION : prédire nbmr_members_season
    # ❌ Retirés : leak_rate (leakage), retained_members_count (leakage),
    #             variation_effectif (calculé depuis la target)
    regression_features = [
        'fk_type_unite',
        'nmbr_members_previous_season',
        'season_year'
    ]

    # ✅ CLASSIFICATION : prédire dropout_risk (basé sur leak_rate)
    # ❌ Retirés : leak_rate (définit le target), taux_retention (dérivé de leak_rate),
    #             variation_effectif (dérivé de la target régression),
    #             retained_members_count (corrélé à leak_rate)
    classification_features = [
        'fk_type_unite',
        'nbmr_members_season',
        'nmbr_members_previous_season',
        'season_year'
    ]

    # ✅ CLUSTERING : grouper les unités par comportement
    clustering_features = [
        'nbmr_members_season',
        'nmbr_members_previous_season',
        'leak_rate'
    ]

    print(f"\n📊 Regression features    ({len(regression_features)}): {regression_features}")
    print(f"📊 Classification features({len(classification_features)}): {classification_features}")
    print(f"📊 Clustering features    ({len(clustering_features)}): {clustering_features}")

    return regression_features, classification_features, clustering_features

def save_datasets(df, reg_feat, class_feat, cluster_feat):
    os.makedirs('data', exist_ok=True)

    df_reg   = df[reg_feat   + ['nbmr_members_season']].copy()
    df_class = df[class_feat + ['dropout_risk']].copy()
    df_clust = df[cluster_feat + ['fk_type_unite', 'season']].copy()

    df_reg.to_csv('data/dataset_regression.csv',     index=False)
    df_class.to_csv('data/dataset_classification.csv', index=False)
    df_clust.to_csv('data/dataset_clustering.csv',   index=False)

    # Dataset série temporelle
    df_ts = df[['fk_type_unite', 'season', 'season_year', 'nbmr_members_season']]\
              .sort_values(['fk_type_unite', 'season_year'])
    df_ts.to_csv('data/dataset_timeseries.csv', index=False)

    print(f"\n💾 Datasets sauvegardés :")
    print(f"   dataset_regression.csv     → {len(df_reg)} lignes")
    print(f"   dataset_classification.csv → {len(df_class)} lignes")
    print(f"   dataset_clustering.csv     → {len(df_clust)} lignes")
    print(f"   dataset_timeseries.csv     → {len(df_ts)} lignes")

    return df_reg, df_class, df_clust

if __name__ == "__main__":
    df = load_data()
    print(f"✅ Données chargées: {len(df)} lignes")
    df = data_cleaning(df)
    df = feature_engineering(df)
    reg_feat, class_feat, cluster_feat = feature_selection(df)
    save_datasets(df, reg_feat, class_feat, cluster_feat)
    print("\n" + "="*60)
    print("✅ DATA PREPARATION TERMINÉE")
    print("="*60)