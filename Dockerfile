FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api/ ./api/
COPY data/processed/chroma_db ./data/processed/chroma_db
COPY data/processed/metriques_ragas.csv ./data/processed/metriques_ragas.csv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
