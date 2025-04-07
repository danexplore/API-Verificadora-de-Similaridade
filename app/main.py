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

# Configuração do Pipefy
PIPEFY_API_URL = "https://api.pipefy.com/graphql"
PIPEFY_API_TOKEN = os.getenv('PIPEFY_API_TOKEN')

ELASTIC_URL_TOKEN = os.getenv("ELASTIC_URL_TOKEN")

# Configuração do Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL = f"https://daniel-elasticsearch.ekyhxs.easypanel.host"

# Inicializar cliente do Elasticsearch
client = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(os.getenv('ELASTIC_USERNAME'), os.getenv('ELASTIC_PASSWORD'))
)


# Inicializar modelo de embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/app/models')

# Inicializar FastAPI
app = FastAPI(title="API de Similaridade de Cursos", version="1.0")

@app.get("/")
async def home():
    return {"message": "API de Similaridade de Cursos Online!"}

@app.get("/buscar/")
async def buscar_similaridade(nome: str, card_id: str = None, resumo: str = None, situacao: str = None, versao: str = None, coordenador: str = None):
    """
    Busca cursos similares no Elasticsearch usando nome e resumo do curso com busca híbrida (texto + vetor).
    """
    try:
        if not nome:
            raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

        else:
            nome_preparado = preparar_para_embedding(nome)
            resumo_preparado = preparar_para_embedding(resumo) if resumo else None
            nome_vector = model.encode(nome_preparado).tolist()

            # Construir filtros dinamicamente
            filters = []
            if situacao:
                filters.append({"term": {"situacao": situacao}})
            if versao:
                filters.append({"term": {"versao": versao}})
            if coordenador:
                filters.append({
                    "match_phrase_prefix": {
                        "coordenador": {
                            "query": coordenador
                        }
                    }
                })

            # Montar blocos do should dinamicamente
            should_clauses = [
                {
                    "knn": {
                        "field": "nome_vector",
                        "query_vector": nome_vector,
                        "k": 15,
                        "num_candidates": 100
                    }
                }
            ]

            # Se tiver resumo, adiciona KNN do resumo_vector
            if resumo_preparado:
                resumo_vector = model.encode(resumo_preparado).tolist()
                should_clauses.append({
                    "knn": {
                        "field": "resumo_vector",
                        "query_vector": resumo_vector,
                        "k": 15,
                        "num_candidates": 100
                    }
                })

            # Montar a query final
            query = {
                "size": 15,
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "filter": filters,
                        "minimum_should_match": 1  # garante que pelo menos 1 `should` case
                    }
                },
                "_source": ["nome", "coordenador", "situacao", "versao"]
            }



        # Fazer a busca
        response = client.search(index="cursos_producao", body=query)
        resultados = response["hits"]["hits"]

        if not resultados:
            return {"message": "Nenhum curso similar encontrado."}

        # Processar resultados com limiar mínimo de similaridade (ex: 60%)
        cursos_similares = ["🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n"]
        for curso in resultados:
            score_percentual = round(((curso['_score'] + 1) / 2.0) * 85, 0)
            if score_percentual < 60:
                continue  # Ignora cursos com baixa similaridade

            cursos_similares.append(
                f"📌 Curso Similar: {curso['_source']['nome']}\n"
                f"📊 Similaridade: {score_percentual}%\n"
                f"👨‍🏫 Coordenador: {curso['_source']['coordenador']}\n"
                f"📌 Situação: {curso['_source']['situacao']}\n"
                f"🆕 Versão: {curso['_source']['versao']}\n"
                f"--------------------------------------------------\n"
            )

        if len(cursos_similares) == 1:
            return {"message": "Nenhum curso similar relevante encontrado."}

        cursos_similares_str = "\n".join(cursos_similares)

        # Atualizar no Pipefy ou retornar
        if card_id is None:
            return {"message": cursos_similares}
        else:
            mutation = """
            mutation {
                updateCardField(input: {
                    card_id: "%s",
                    field_id: "cursos_similares",
                    new_value: "%s"
                }) {
                    card { id }
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