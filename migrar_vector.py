import pandas as pd
import requests
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL")
VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")

df = pd.read_excel("cursos-elastic-search.xlsx")
df = df.where(pd.notnull(df), "")

headers = {
    "Authorization": f"Bearer {VECTOR_TOKEN}",
    "Content-Type": "application/json"
}

for _, row in df.iterrows():
    payload = {
        "id": str(row["ID"]),
        "data": {"nome": row["Nome do Curso"]}
    }

    response = requests.post(VECTOR_URL, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print(f"Erro ao inserir {row['ID']}: {response.text}")
print("Migração de embeddings para o Upstash Vector concluída.") 