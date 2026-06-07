
import streamlit as st
import requests
import json
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title = "Biomedical RAG : Mental Health",
    page_icon  = "🧠",
    layout     = "wide"
)

API_URL = "http://127.0.0.1:8000"

# ============================================================
# Fonctions utilitaires
# ============================================================

def query_api(question, k=5):
    """Envoie une question à l'API RAG."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json    = {"question": question, "k": k},
            timeout = 60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}

def get_metrics():
    """Récupère les métriques RAGAS."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_health():
    """Vérifie la santé de l'API."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=30)
        return response.status_code == 200
    except:
        return False

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🧠 Biomedical RAG")
st.sidebar.markdown("**Domaine** : Santé mentale des jeunes diplômés")
st.sidebar.markdown("**Modèle** : Llama 3.3 70B via Groq")
st.sidebar.markdown("**Embedding** : all-MiniLM-L6-v2")
st.sidebar.markdown("**Base** : 836 chunks PubMed")

st.sidebar.divider()

# Statut API
api_ok = get_health()
if api_ok:
    st.sidebar.success("✅ API connectée")
else:
    st.sidebar.error("❌ API non connectée")

st.sidebar.divider()

# Métriques RAGAS
metrics = get_metrics()
if metrics:
    st.sidebar.markdown("### 📊 Métriques RAGAS")
    st.sidebar.metric("Context Relevance",  f"{metrics['context_relevance']:.3f}")
    st.sidebar.metric("Faithfulness",       f"{metrics['faithfulness']:.3f}")
    st.sidebar.metric("Answer Relevancy",   f"{metrics['answer_relevancy']:.3f}")
    st.sidebar.metric("Score Global",       f"{metrics['global_score']:.3f}")

st.sidebar.divider()

# Paramètres
st.sidebar.markdown("### ⚙️ Paramètres")
k_chunks = st.sidebar.slider(
    "Nombre de chunks récupérés",
    min_value=3, max_value=10, value=5
)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["💬 Question Answering", "📊 Évaluation du système", "📚 À propos"]
)

# ============================================================
# PAGE 1 : Question Answering
# ============================================================

if page == "💬 Question Answering":
    st.title("💬 Biomedical Question Answering")
    st.markdown("""
    Posez une question sur la **santé mentale des jeunes diplômés** 
    en recherche d'emploi. Le système récupère les articles PubMed 
    les plus pertinents et génère une réponse sourcée.
    """)

    # Exemples de questions
    st.markdown("### 💡 Exemples de questions")
    exemples = [
        "What are the main mental health consequences of unemployment in young graduates?",
        "Does job insecurity cause depression in recent graduates?",
        "What is the scarring effect of unemployment on mental health?",
        "How does precarious employment affect wellbeing of educated workers?",
        "What interventions help improve mental health of unemployed graduates?"
    ]

    col1, col2 = st.columns(2)
    for i, exemple in enumerate(exemples):
        if i % 2 == 0:
            if col1.button(f"📌 {exemple[:60]}...", key=f"ex_{i}"):
                st.session_state['question'] = exemple
        else:
            if col2.button(f"📌 {exemple[:60]}...", key=f"ex_{i}"):
                st.session_state['question'] = exemple

    st.divider()

    # Zone de question
    question = st.text_area(
        "Votre question :",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="Ex: What are the psychological effects of post-graduation unemployment?"
    )

    if st.button("🔍 Rechercher", type="primary", disabled=not api_ok):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                resultat = query_api(question, k=k_chunks)

            if "error" in resultat:
                st.error(f"Erreur : {resultat['error']}")
            else:
                # Mode badge
                if resultat['mode'] == 'strict':
                    st.success(f"✅ Mode Strict — Score similarité : {resultat['score_moyen']:.4f}")
                else:
                    st.warning(f"⚠️ Mode Fallback — Score similarité : {resultat['score_moyen']:.4f}")

                # Réponse
                st.markdown("### 📝 Réponse")
                st.markdown(resultat['answer'])

                st.divider()

                # Sources
                st.markdown("### 📚 Sources PubMed utilisées")
                for i, source in enumerate(resultat['sources']):
                    with st.expander(f"[{i+1}] PMID {source['pmid']} ({source['annee']}) — Score : {source['score']:.4f}"):
                        st.markdown(f"**Titre** : {source['titre']}")
                        st.markdown(f"**Journal** : {source['journal']}")
                        st.markdown(f"**Lien** : [{source['url_pubmed']}]({source['url_pubmed']})")
        else:
            st.warning("Veuillez entrer une question.")

# ============================================================
# PAGE 2 : Évaluation
# ============================================================

elif page == "📊 Évaluation du système":
    st.title("📊 Évaluation du Système RAG")
    st.markdown("Résultats de l'évaluation sur **16 questions** en mode Strict avec des métriques RAGAS-style.")

    metrics = get_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Context Relevance", f"{metrics['context_relevance']:.3f}", delta="✅ > 0.7")
        col2.metric("Faithfulness", f"{metrics['faithfulness']:.3f}", delta="⚠️ Extrapolations LLM")
        col3.metric("Answer Relevancy", f"{metrics['answer_relevancy']:.3f}", delta="✅ > 0.7")
        col4.metric("Score Global", f"{metrics['global_score']:.3f}", delta="✅ Bon système RAG")

        st.divider()
        st.markdown("""
### 🔍 Interprétation

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| Context Relevance | 0.837 | ✅ Les chunks récupérés sont très pertinents |
| Faithfulness | 0.675 | ⚠️ Le LLM enrichit avec ses connaissances générales |
| Answer Relevancy | 0.863 | ✅ Les réponses répondent précisément aux questions |
| **Score Global** | **0.792** | ✅ **Bon système RAG** |

### 📋 Détail par mode
- **Mode Strict** : 16/23 questions → contexte suffisant
- **Mode Fallback** : 7/23 questions → questions hors domaine
        """)
    else:
        st.error("Impossible de récupérer les métriques : vérifiez que l'API est connectée.")

# ============================================================
# PAGE 3 : À propos
# ============================================================

elif page == "📚 À propos":
    st.title("📚 À propos du projet")
    st.markdown("""
    ## Biomedical Question Answering with RAG and LLMs

    ### 🎯 Objectif
    Système de question-réponse biomédical spécialisé sur la 
    **santé mentale des jeunes diplômés en recherche d'emploi**.

    ### 🏗️ Architecture
    - **652 abstracts PubMed** collectés via API (2000-2025)
    - **836 chunks** indexés dans ChromaDB
    - **Embeddings** : all-MiniLM-L6-v2 (384 dimensions)
    - **LLM** : Llama 3.3 70B via Groq API
    - **Stratégie** : RAG avec fallback (mode strict / mode chain-of-thought)

    ### 📊 Performances
    | Métrique | Score |
    |----------|-------|
    | Context Relevance | 0.837 |
    | Faithfulness | 0.675 |
    | Answer Relevancy | 0.863 |
    | **Score Global** | **0.792** |

    ### 🔗 Liens
    - **GitHub** : [Dboy003/biomedical-rag-llm](https://github.com/Dboy003/biomedical-rag-llm)
    - **API Docs** : [FastAPI Swagger](http://127.0.0.1:8000/docs)

    ### 👤 Auteur
    **Mourad DO-REGO** : Data Scientist / Biostatisticien
    """)
