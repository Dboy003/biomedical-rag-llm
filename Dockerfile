FROM python:3.11-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt
RUN pip install --no-cache-dir streamlit requests

# Copie des fichiers
COPY api/ ./api/
COPY app.py .
COPY data/processed/chroma_db ./data/processed/chroma_db
COPY data/processed/metriques_ragas.csv ./data/processed/metriques_ragas.csv

# Script de démarrage
COPY start.sh .
RUN chmod +x start.sh

# Port HuggingFace Spaces
EXPOSE 7860

CMD ["./start.sh"]
