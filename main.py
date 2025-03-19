from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env
load_dotenv()

# Configuração do Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL = "https://6989d0bb119d4c8ab118c2b7113a0d31.southamerica-east1.gcp.elastic-cloud.com:443"
ENCODED_CREDENTIALS = os.getenv('encoded')

# Inicializar cliente do Elasticsearch
client = Elasticsearch(
    ELASTICSEARCH_URL,
    api_key=ENCODED_CREDENTIALS
)

# Inicializar modelo de embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Inicializar FastAPI
app = FastAPI(title="API de Similaridade de Cursos", version="1.0")

@app.get("/")
def home():
    return {"message": "API de Similaridade de Cursos Online!"}

@app.get("/buscar/")
def buscar_similaridade(nome: str):
    """
    Busca cursos similares no Elasticsearch usando apenas o nome do curso.
    """
    try:
        # Gerar embedding do nome do curso
        query_vector = model.encode(nome).tolist()

        # Criar a query para busca por similaridade no Elasticsearch
        query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        # Fazer a busca no Elasticsearch
        response = client.search(index="cursos", body=query)
        resultados = response["hits"]["hits"]

        if not resultados:
            return {"message": "Nenhum curso similar encontrado."}

        # Processar os resultados
        cursos_similares = []
        for curso in resultados:
            cursos_similares.append(
                f"--------------------------------------------------\n"
                f"📌 Curso Similar: {curso['_source']['nome']}\n"
                f"📊 Similaridade: {round((curso['_score'] / 2.0) * 100, 2)}%\n"
                f"👨‍🏫 Coordenador: {curso['_source']['coordenador']}\n"
                f"📌 Situação: {curso['_source']['situacao']}\n"
                f"🆕 Versão: {curso['_source']['versao']}"
            )

        return {"cursos_similares": cursos_similares}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {str(e)}")
