from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import requests
import unicodedata
import re

def preparar_para_embedding(texto: str) -> str:
    # Remover acentos
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    # Remover símbolos que não ajudam semanticamente
    texto = re.sub(r"[\[\]\(\)\:\-\_]", " ", texto)
    # Remover múltiplos espaços e deixar minúsculo
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto

# Carregar variáveis de ambiente do .env
load_dotenv()

ELASTIC_URL_TOKEN = os.getenv("ELASTIC_URL_TOKEN")

# Configuração do Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL = f"https://{ELASTIC_URL_TOKEN}.southamerica-east1.gcp.elastic-cloud.com:443"
ENCODED_CREDENTIALS = os.getenv('encoded')

# Configuração do Pipefy
PIPEFY_API_URL = "https://api.pipefy.com/graphql"
PIPEFY_API_TOKEN = os.getenv('PIPEFY_API_TOKEN')

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
async def home():
    return {"message": "API de Similaridade de Cursos Online!"}

@app.get("/buscar/")
async def buscar_similaridade(nome: str, card_id: str):
    """
    Busca cursos similares no Elasticsearch usando apenas o nome do curso e atualiza o campo do cartão no Pipefy.
    """
    try:
        if not nome or not card_id:
            raise HTTPException(status_code=400, detail="Nome do curso e ID do cartão são obrigatórios.")

        # Gerar embedding do nome do curso
        query_vector = model.encode(preparar_para_embedding(nome)).tolist()

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
        cursos_similares.append("🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n")

        for curso in resultados:
            cursos_similares.append(
                f"📌 Curso Similar: {curso['_source']['nome']}\n"
                f"📊 Similaridade: {round((curso['_score'] / 2.0) * 100, 2)}%\n"
                f"👨‍🏫 Coordenador: {curso['_source']['coordenador']}\n"
                f"📌 Situação: {curso['_source']['situacao']}\n"
                f"🆕 Versão: {curso['_source']['versao']}\n"
                f"--------------------------------------------------\n"
            )

        cursos_similares_str = "\n".join(cursos_similares)

        # Atualizar o campo do cartão no Pipefy
        mutation = """
        mutation {
            updateCardField(input: {
                card_id: "%s",
                field_id: "cursos_similares",
                new_value: "%s"
            }) {
                card {
                    id
                }
            }
        }
        """ % (card_id, cursos_similares_str)

        headers = {
            "Authorization": f"Bearer {PIPEFY_API_TOKEN}",
            "Content-Type": "application/json"
        }

        pipefy_response = requests.post(PIPEFY_API_URL, json={"query": mutation}, headers=headers)

        if pipefy_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Erro ao atualizar o campo do cartão no Pipefy.")

        return {"message": "Campo do cartão atualizado com sucesso.", "cursos_similares": cursos_similares_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {str(e)}")
    