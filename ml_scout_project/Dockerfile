# =============================================================================
# Dockerfile — ML Scout API
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY flask_api.py .
COPY 02_regression.py .
COPY 03_classification.py .
COPY 04_clustering.py .
COPY 06_anomaly_detection.py .
COPY 07_mlops_mlflow.py .
COPY 08_mlops_registry.py .
COPY 09_mlops_staging_production.py .
COPY build_features.py .
COPY data/ ./data/
COPY models/ ./models/
COPY visuals/ ./visuals/

# Exposer le port
EXPOSE 5000

# Lancer l'API
CMD ["python", "flask_api.py"]
