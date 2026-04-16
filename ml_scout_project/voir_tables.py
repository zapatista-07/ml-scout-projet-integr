# voir_tables.py
from sqlalchemy import create_engine, text
import pandas as pd
from config import get_connection_string

print("🔌 Connexion à la base...")
engine = create_engine(get_connection_string())

# Lister TOUTES les tables
print("\n" + "="*60)
print("📋 TOUTES LES TABLES DANS LA BASE :")
print("="*60)

query = text("""
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
ORDER BY table_name;
""")

tables = pd.read_sql_query(query, engine)
print(tables.to_string(index=False))

print(f"\n✅ Total: {len(tables)} tables")

# Vérifier le contenu de chaque table
print("\n" + "="*60)
print("📊 CONTENU DE CHAQUE TABLE :")
print("="*60)

for _, row in tables.iterrows():
    table_name = row['table_name']
    count_query = text(f'SELECT COUNT(*) as nb FROM "{table_name}"')
    try:
        count_df = pd.read_sql_query(count_query, engine)
        nb = count_df['nb'].values[0]
        
        # Afficher les premières colonnes
        columns_query = text(f'SELECT * FROM "{table_name}" LIMIT 1')
        columns_df = pd.read_sql_query(columns_query, engine)
        cols = ', '.join(columns_df.columns.tolist())
        
        print(f"\n📁 {table_name}:")
        print(f"   - Lignes: {nb}")
        print(f"   - Colonnes: {cols}")
    except Exception as e:
        print(f"\n❌ {table_name}: Erreur - {str(e)[:50]}")

engine.dispose()