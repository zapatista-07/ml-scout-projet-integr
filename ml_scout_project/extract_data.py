# extract_data.py
# Extraction des données pour prédiction du leak_rate

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import get_connection_string

def extract_ml_dataset():
    """
    Extrait les données depuis fact_registrations
    pour créer le dataset de prédiction du dropout
    """
    print("🔌 Connexion à DWH_scout...")
    engine = create_engine(get_connection_string())
    
    # ─────────────────────────────────────────────────────────────
    # Charger fact_registrations
    # ─────────────────────────────────────────────────────────────
    print("📥 Chargement de fact_registrations...")
    
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
    print(f"✅ fact_registrations: {len(df)} lignes")
    
    # ─────────────────────────────────────────────────────────────
    # Nettoyer les données
    # ─────────────────────────────────────────────────────────────
    print("\n🔧 Nettoyage...")
    df = df.fillna(0)
    
    # ─────────────────────────────────────────────────────────────
    # Créer des features additionnelles
    # ─────────────────────────────────────────────────────────────
    print("🎯 Création des features...")
    
    # Variation des effectifs
    df['variation_effectif'] = (
        (df['nbmr_members_season'] - df['nmbr_members_previous_season']) / 
        df['nmbr_members_previous_season'].replace(0, np.nan) * 100
    ).round(2)
    
    # Taux de rétention
    df['taux_retention'] = (
        df['retained_members_count'] / 
        df['nmbr_members_previous_season'].replace(0, np.nan) * 100
    ).round(2)
    
    # Perte absolue
    df['perte_membres'] = (
        df['nmbr_members_previous_season'] - df['retained_members_count']
    )
    
    df = df.fillna(0)
    
    # ─────────────────────────────────────────────────────────────
    # Afficher statistiques
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("📊 STATISTIQUES")
    print("="*60)
    print(f"Lignes: {len(df)}")
    print(f"\nDistribution leak_rate:")
    print(df['leak_rate'].describe())
    
    # ─────────────────────────────────────────────────────────────
    # Sauvegarder
    # ─────────────────────────────────────────────────────────────
    df.to_csv('data/ml_dataset_final.csv', index=False)
    print(f"\n💾 Dataset sauvegardé: data/ml_dataset_final.csv")
    
    engine.dispose()
    return df

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    df = extract_ml_dataset()
    
    print("\n" + "="*60)
    print("📋 APERÇU")
    print("="*60)
    print(df.head().to_string())