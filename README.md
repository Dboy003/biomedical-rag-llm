# 🧠 Biomedical Question Answering with RAG and LLMs

> Système de question-réponse biomédical spécialisé sur la **santé mentale des jeunes diplômés en recherche d'emploi**, basé sur 652 abstracts PubMed (2000-2025).

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://www.trychroma.com/)

---

## Contexte

Dans ce projet, j'ai construit un assistant de recherche biomédical capable de répondre à des questions sur l'impact du chômage post-diplôme sur la santé mentale des jeunes diplômés universitaires.

Il s'inscrit dans la préparation d'une **revue systématique et méta-analyse** sur ce sujet, le système RAG sert d'outil d'exploration de la littérature scientifique.

---

## Architecture du projet

```text
biomedical-rag-llm/
├── notebooks/
│   ├── 01_data_collection.ipynb    # Collecte PubMed API (652 articles)
│   ├── 02_indexing.ipynb           # Chunking + embeddings + ChromaDB
│   ├── 03_rag_pipeline.ipynb       # Pipeline RAG avec fallback
│   ├── 04_evaluation.ipynb         # Évaluation RAGAS-style
│   └── 05_api_dashboard.ipynb      # API FastAPI + Dashboard Streamlit
├── api/
│   └── main.py                     # API REST FastAPI
├── data/
│   ├── raw/                        # Données brutes (non versionnées)
│   └── processed/                  # ChromaDB + métriques
├── app.py                          # Dashboard Streamlit
└── requirements.txt
```

---

## Dataset

- **Source** : API PubMed (Entrez/Biopython)
- **Période** : 2000 à 2025
- **Volume** : 652 abstracts uniques avec 836 chunks indexés
- **Thématiques** :
  - Santé mentale des jeunes diplômés au chômage
  - Transition post-diplôme et santé mentale
  - Emploi précaire et détresse psychologique
  - Revues systématiques sur le sujet

---

## Démo déployé sur Render

**Base URL** : https://biomedical-rag-api.onrender.com

---

## Pipeline RAG

```text
Question utilisateur
↓
Embedding (all-MiniLM-L6-v2)
↓
Recherche ChromaDB (Top 5 chunks)
↓
Score de pertinence < 0.80 ?
↙              ↘
OUI              NON
↓                ↓
Mode Strict      Mode Fallback
(réponse         (chain-of-thought
sourcée)         + avertissement)
↘              ↙
Réponse + Sources PubMed citées
```
---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | all-MiniLM-L6-v2 (384 dim) |
| Vector DB | ChromaDB |
| RAG Framework | LangChain |
| API | FastAPI |
| Dashboard | Streamlit |
| Données | PubMed API (Biopython/Entrez) |

---

## Évaluation RAGAS

Évaluation sur **16 questions** en mode Strict :

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| Context Relevance | **0.837** | ✅ Chunks très pertinents |
| Faithfulness | **0.675** | ⚠️ Légères extrapolations LLM |
| Answer Relevancy | **0.863** | ✅ Réponses précises |
| **Score Global** | **0.792** | ✅ **Bon système RAG** |

---

## API

**Base URL** : `http://127.0.0.1:8000`

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Statut de l'API |
| `/health` | GET | Santé du système |
| `/query` | POST | Question → Réponse RAG |
| `/metrics` | GET | Métriques RAGAS |

### Exemple de requête

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the mental health consequences of unemployment in young graduates?",
    "k": 5
  }'
```

### Exemple de réponse

```json
{
  "question": "What are the mental health consequences of unemployment in young graduates?",
  "answer": "**Main Finding**: Unemployment in young graduates is associated with...",
  "mode": "strict",
  "score_moyen": 0.6706,
  "sources": [
    {
      "pmid": "33359536",
      "titre": "Effects of graduating during economic downturns on mental health",
      "journal": "Annals of epidemiology",
      "annee": "2021",
      "score": 0.6284,
      "url_pubmed": "https://pubmed.ncbi.nlm.nih.gov/33359536/"
    }
  ],
  "n_chunks": 5
}
```

---

## Installation locale

```bash
# Cloner le repo
git clone https://github.com/Dboy003/biomedical-rag-llm.git
cd biomedical-rag-llm

# Créer l'environnement virtuel
py -3.11 -m venv venv
.\venv\Scripts\Activate

# Installer les dépendances
pip install -r requirements.txt

# Configurer les clés API
echo "GROQ_API_KEY=ta_clé_groq" > .env
echo "PUBMED_EMAIL=ton_email" >> .env

# Lancer l'API
uvicorn api.main:app --reload --port 8000

# Lancer le dashboard (nouveau terminal)
streamlit run app.py
```

---

## Perspectives

- **Fine-tuning** d'un LLM médical (Mistral/Llama) sur des données annotées
- **Connexion temps réel** à PubMed pour indexer les nouveaux articles
- **Déploiement cloud** sur Hugging Face Spaces
- **Intégration** dans le workflow de ma méta-analyse : Impact de la transition post-diplôme et du chômage initial sur la santé mentale des jeunes diplômés universitaires. (Le projet qui m'a inspiré ce projet).

---

## 👤 Auteur

**Mourad DO-REGO** : [GitHub](https://github.com/Dboy003)