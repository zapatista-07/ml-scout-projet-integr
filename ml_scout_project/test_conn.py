# test_conn.py
from sqlalchemy import create_engine, text
from config import get_connection_string

try:
    print("🔌 Connexion à DWH_scout...")
    engine = create_engine(get_connection_string())
    
    with engine.connect() as conn:
        # Utiliser text() pour les requêtes SQL brutes (SQLAlchemy 2.0+)
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print(f"✅ CONNEXION RÉUSSIE !")
        print(f"📊 PostgreSQL: {version[:50]}...")
    
    engine.dispose()
    
except Exception as e:
    print(f"❌ ÉCHEC: {e}")
    print("\n💡 Vérifie:")
    print("   1. PostgreSQL est démarré")
    print("   2. La base 'DWH_scout' existe")
    print("   3. Le mot de passe dans config.py est correct")