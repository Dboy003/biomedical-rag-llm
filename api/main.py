
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import time
import json

# LangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# Initialisation
# ============================================================

app = FastAPI(
    title       = "Biomedical RAG API",
    description = "Question Answering system on mental health of young graduates using RAG and LLMs",
    version     = "1.0.0"
)

# Chemins
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "processed"
CHROMA_DIR = str(DATA_DIR / "chroma_db")

# Chargement clé API
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Chargement des composants
print("Chargement des composants RAG...")

embedding_model = HuggingFaceEmbeddings(
    model_name    = "all-MiniLM-L6-v2",
    model_kwargs  = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory  = CHROMA_DIR,
    embedding_function = embedding_model,
    collection_name    = "pubmed_biomedical"
)

llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    temperature = 0.1,
    api_key     = GROQ_API_KEY
)

# Paramètres RAG
SEUIL_PERTINENCE = 0.80
K_CHUNKS         = 5
MARGE_IC         = 0.05

# Prompts
PROMPT_STRICT = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized biomedical research assistant 
focused on mental health of young graduates and unemployment.

STRICT RULES:
1. Answer ONLY based on the provided PubMed context below
2. NEVER invent information not present in the context
3. Always cite the PMID and year of the sources you use
4. Structure your answer with: Main Finding, Evidence, and Limitations

CONTEXT FROM PUBMED:
{context}
"""),
    ("human", "Question: {question}\n\nProvide an evidence-based answer.")
])

PROMPT_FALLBACK = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized biomedical research assistant.

IMPORTANT: The available PubMed context has limited relevance.
Use chain-of-thought reasoning but MUST clearly warn the user.

RULES:
1. Start with: " LIMITED CONTEXT WARNING: ..."
2. Use step-by-step reasoning
3. Recommend consulting additional sources

AVAILABLE CONTEXT:
{context}
"""),
    ("human", "Question: {question}\n\nReason step-by-step with appropriate caveats.")
])

# ============================================================
# Schémas
# ============================================================

class QueryInput(BaseModel):
    question : str
    k        : Optional[int] = 5

class Source(BaseModel):
    pmid      : str
    titre     : str
    journal   : str
    annee     : str
    score     : float
    url_pubmed: str

class QueryOutput(BaseModel):
    question    : str
    answer      : str
    mode        : str
    score_moyen : float
    sources     : List[Source]
    n_chunks    : int

# ============================================================
# Fonctions RAG
# ============================================================

def formater_contexte(docs_scores):
    parts = []
    for i, (doc, score) in enumerate(docs_scores):
        parts.append(f"""[Source {i+1}]
PMID    : {doc.metadata["pmid"]}
Titre   : {doc.metadata["titre"]}
Année   : {doc.metadata["annee"]}
Score   : {score:.4f}
Contenu : {doc.page_content}""")
    return "\n---\n".join(parts)

def extraire_sources(docs_scores):
    sources = []
    for doc, score in docs_scores:
        sources.append(Source(
            pmid       = doc.metadata["pmid"],
            titre      = doc.metadata["titre"],
            journal    = doc.metadata["journal"],
            annee      = doc.metadata["annee"],
            score      = round(score, 4),
            url_pubmed = f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata['pmid']}/"
        ))
    return sources

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "status"     : "online",
        "model"      : "Llama 3.3 70B via Groq",
        "embedding"  : "all-MiniLM-L6-v2",
        "chunks"     : vectorstore._collection.count(),
        "description": "Biomedical RAG : Mental health of young graduates"
    }

@app.get("/health")
def health():
    try:
        test = vectorstore.similarity_search("test", k=1)
        return {
            "status"        : "healthy",
            "vectorstore"   : "connected",
            "chunks"        : vectorstore._collection.count(),
            "llm"           : "Llama 3.3 70B via Groq"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryOutput)
def query(data: QueryInput):
    try:
        # Recherche
        docs_scores = vectorstore.similarity_search_with_score(
            query=data.question, k=data.k
        )
        scores      = [score for _, score in docs_scores]
        score_moyen = sum(scores) / len(scores)

        # Choix du mode
        if score_moyen < SEUIL_PERTINENCE:
            mode   = "strict"
            prompt = PROMPT_STRICT
        else:
            mode   = "fallback"
            prompt = PROMPT_FALLBACK

        # Génération
        contexte = formater_contexte(docs_scores)
        chain    = prompt | llm | StrOutputParser()
        answer   = chain.invoke({
            "question": data.question,
            "context" : contexte
        })

        return QueryOutput(
            question    = data.question,
            answer      = answer,
            mode        = mode,
            score_moyen = round(score_moyen, 4),
            sources     = extraire_sources(docs_scores),
            n_chunks    = len(docs_scores)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    metrics_path = DATA_DIR / "metriques_ragas.csv"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")

    import pandas as pd
    df = pd.read_csv(metrics_path)

    return {
        "n_questions"       : len(df),
        "context_relevance" : round(df["context_relevance"].mean(), 3),
        "faithfulness"      : round(df["faithfulness"].mean(), 3),
        "answer_relevancy"  : round(df["answer_relevancy"].mean(), 3),
        "global_score"      : round(df[["context_relevance",
                                        "faithfulness",
                                        "answer_relevancy"]].mean().mean(), 3)
    }
