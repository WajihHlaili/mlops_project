# Utiliser une image Python officielle version 3.13
FROM python:3.13-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Flask
EXPOSE 85

# Lancer à la fois Flask et FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"] && python app.py
