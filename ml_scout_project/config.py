# config.py
# Configuration de la base de données

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'DWH_scout',
    'user': 'postgres',
    'password': 'postgres123'  
}

def get_connection_string():
    """Retourne la chaîne de connexion SQLAlchemy"""
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Features pour le modèle ML
ML_FEATURES = [
    'adherents',
    'chefs', 
    'nb_activites',
    'budget_total_dt',
    'ratio_chef_membre',
    'budget_par_activite',
    'total_participants_camps',
    'nb_camps',
    'duree_moyenne_camps',
    'budget_total_camps'
]

TARGET_COLUMN = 'dropout_target'