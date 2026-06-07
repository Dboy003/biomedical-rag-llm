#!/bin/bash

# Lancement de l'API FastAPI en arrière-plan
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Attente que l'API soit prête
sleep 10

# Lancement du dashboard Streamlit sur le port HF Spaces
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
