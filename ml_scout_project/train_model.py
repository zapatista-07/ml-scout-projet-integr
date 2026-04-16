# train_model.py
# Entraînement du modèle de prédiction du leak_rate

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_dataset(filepath='data/ml_dataset_final.csv'):
    """Charge le dataset"""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset chargé: {len(df)} lignes")
    return df

def prepare_features(df):
    """Prépare les features et target"""
    
    # Features (sans leak_rate et sans retained_members_count)
    features = [
        'fk_type_unite',
        'nbmr_members_season',
        'nmbr_members_previous_season',
        'variation_effectif',
        'taux_retention',
        'perte_membres'
    ]
    
    X = df[features].copy()
    y = df['leak_rate'].copy()  # TARGET
    
    print(f"Features: {features}")
    print(f"Target: leak_rate")
    
    return X, y, features

def train_model(X, y, features):
    """Entraîne le modèle avec Leave-One-Out CV"""
    
    print("\n" + "="*60)
    print("🤖 ENTRAÎNEMENT DU MODÈLE")
    print("="*60)
    
    # Modèle: Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,  # Limite pour éviter overfitting
        random_state=42,
        n_jobs=-1
    )
    
    # Leave-One-Out Cross-Validation
    print("\n📊 Validation Leave-One-Out...")
    cv_scores = cross_val_score(
        model, X, y, 
        cv=LeaveOneOut(), 
        scoring='neg_mean_absolute_error'
    )
    
    mae_cv = -cv_scores.mean()
    std_cv = cv_scores.std()
    
    print(f"MAE moyen (CV): {mae_cv:.2f}%")
    print(f"Std: {std_cv:.2f}%")
    
    # Entraînement sur toutes les données
    print("\n🔧 Entraînement final...")
    model.fit(X, y)
    
    # Prédictions
    y_pred = model.predict(X)
    
    # Metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE FINALE")
    print("="*60)
    print(f"MAE: {mae:.2f}%")
    print(f"RMSE: {rmse:.2f}%")
    print(f"R²: {r2:.3f}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔝 Feature Importance:")
    print(feature_importance.to_string(index=False))
    
    return model, feature_importance, mae, rmse, r2

def save_model(model, feature_importance, metrics, filepath='models/'):
    """Sauvegarde le modèle"""
    import os
    os.makedirs(filepath, exist_ok=True)
    
    # Sauvegarder le modèle
    joblib.dump(model, f'{filepath}dropout_predictor.pkl')
    print(f"\n💾 Modèle sauvegardé: {filepath}dropout_predictor.pkl")
    
    # Sauvegarder feature importance
    feature_importance.to_csv(f'{filepath}feature_importance.csv', index=False)
    
    # Sauvegarder metrics
    with open(f'{filepath}metrics.txt', 'w') as f:
        f.write(f"MAE: {metrics[0]:.2f}%\n")
        f.write(f"RMSE: {metrics[1]:.2f}%\n")
        f.write(f"R²: {metrics[2]:.3f}\n")
    print(f"💾 Metrics sauvegardées: {filepath}metrics.txt")

if __name__ == "__main__":
    # Charger les données
    df = load_dataset()
    
    # Préparer features/target
    X, y, features = prepare_features(df)
    
    # Entraîner le modèle
    model, feature_importance, mae, rmse, r2 = train_model(X, y, features)
    
    # Sauvegarder
    save_model(model, feature_importance, (mae, rmse, r2))
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*60)