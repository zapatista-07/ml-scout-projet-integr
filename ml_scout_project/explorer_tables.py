# explorer_tables.py
# Explorer TOUTES les tables disponibles dans les deux bases

from sqlalchemy import create_engine, text
import pandas as pd

# Configuration pour les deux bases
bases = {
    'sa_scout (Source)': 'postgresql://postgres:postgres123@localhost:5432/sa_scout',
    'DWH_scout (Warehouse)': 'postgresql://postgres:postgres123@localhost:5432/DWH_scout'
}

print("="*80)
print("🔍 EXPLORATION COMPLÈTE DES BASES DE DONNÉES")
print("="*80)

for nom_base, conn_string in bases.items():
    print(f"\n{'='*80}")
    print(f"📁 BASE: {nom_base}")
    print("="*80)
    
    engine = create_engine(conn_string)
    
    # Lister toutes les tables
    query = text("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name;
    """)
    
    tables = pd.read_sql_query(query, engine)
    
    print(f"\n📋 Tables disponibles ({len(tables)}):")
    print("-"*80)
    
    # Classifier les tables
    dim_tables = [t for t in tables['table_name'] if t.startswith('dim_')]
    fact_tables = [t for t in tables['table_name'] if t.startswith('fact_')]
    autres_tables = [t for t in tables['table_name'] if not t.startswith('dim_') and not t.startswith('fact_')]
    
    if dim_tables:
        print(f"\n📊 DIMENSIONS ({len(dim_tables)}):")
        for table in dim_tables:
            count_query = text(f'SELECT COUNT(*) as nb FROM "{table}"')
            count_df = pd.read_sql_query(count_query, engine)
            nb = count_df['nb'].values[0]
            
            # Afficher les colonnes
            cols_query = text(f"SELECT * FROM \"{table}\" LIMIT 1")
            try:
                cols_df = pd.read_sql_query(cols_query, engine)
                cols = ', '.join(cols_df.columns.tolist())
                print(f"   ✅ {table}: {nb} lignes | Colonnes: {cols[:100]}...")
            except:
                print(f"   ✅ {table}: {nb} lignes")
    
    if fact_tables:
        print(f"\n📈 FACT TABLES ({len(fact_tables)}):")
        for table in fact_tables:
            count_query = text(f'SELECT COUNT(*) as nb FROM "{table}"')
            count_df = pd.read_sql_query(count_query, engine)
            nb = count_df['nb'].values[0]
            
            # Afficher les colonnes
            cols_query = text(f"SELECT * FROM \"{table}\" LIMIT 1")
            try:
                cols_df = pd.read_sql_query(cols_query, engine)
                cols = ', '.join(cols_df.columns.tolist())
                print(f"   ✅ {table}: {nb} lignes | Colonnes: {cols[:100]}...")
            except:
                print(f"   ✅ {table}: {nb} lignes")
    
    if autres_tables:
        print(f"\n📂 AUTRES TABLES ({len(autres_tables)}):")
        for table in autres_tables:
            count_query = text(f'SELECT COUNT(*) as nb FROM "{table}"')
            count_df = pd.read_sql_query(count_query, engine)
            nb = count_df['nb'].values[0]
            print(f"   ✅ {table}: {nb} lignes")
    
    engine.dispose()

print("\n" + "="*80)
print("✅ EXPLORATION TERMINÉE")
print("="*80)