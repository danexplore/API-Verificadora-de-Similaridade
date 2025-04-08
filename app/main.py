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

        nome_preparado = preparar_para_embedding(nome)
        nome_vector = model.encode(nome_preparado).tolist()

        resumo_preparado = preparar_para_embedding(resumo) if resumo else None
        resumo_vector = model.encode(resumo_preparado).tolist() if resumo_preparado else None

        # Filtros comuns para as duas buscas
        filters = []
        if situacao:
            situacoes = [s.strip() for s in situacao.split(",")]
            filters.append({"terms": {"situacao": situacoes}})
        if versao:
            versoes = [v.strip() for v in versao.split(",")]
            filters.append({"terms": {"versao": versoes}})
        if coordenador:
            filters.append({"match_phrase_prefix": {"coordenador": {"query": coordenador}}})

        # Função para montar a query de KNN
        def montar_query_knn(vector_field, vector):
            return {
                "size": 50,
                "query": {
                    "bool": {
                        "filter": filters,
                        "must": {
                            "knn": {
                                "field": vector_field,
                                "query_vector": vector,
                                "k": 75,
                                "num_candidates": 300
                            }
                        }
                    }
                },
                "_source": ["nome", "coordenador", "situacao", "versao"]
            }

        # Executa busca por nome
        query_nome = montar_query_knn("nome_vector", nome_vector)
        res_nome = client.search(index="cursos_producao", body=query_nome)["hits"]["hits"]

        # Executa busca por resumo, se houver
        res_resumo = []
        if resumo_vector:
            query_resumo = montar_query_knn("resumo_vector", resumo_vector)
            res_resumo = client.search(index="cursos_producao", body=query_resumo)["hits"]["hits"]

        # Indexar os scores
        scores_nome = {r["_id"]: r["_score"] for r in res_nome}
        scores_resumo = {r["_id"]: r["_score"] for r in res_resumo}

        # Mesclar e calcular score final
        todos_ids = set(scores_nome.keys()).union(scores_resumo.keys())
        peso_nome = 0.7
        peso_resumo = 0.3

        # Processar resultados com limiar mínimo de similaridade (ex: 60%)
        cursos_similares = ["🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n"]
        cursos_final = []
        for _id in todos_ids:
            score_nome = scores_nome.get(_id, 0)
            score_resumo = scores_resumo.get(_id, 0)
            score_final = (peso_nome * score_nome) + (peso_resumo * score_resumo)
            if score_final < 0.71:
                continue

            # Buscar o documento completo (de qualquer uma das buscas)
            doc = next((r for r in res_nome + res_resumo if r["_id"] == _id), None)
            if not doc:
                continue

            cursos_final.append({
                "nome": doc["_source"]["nome"],
                "coordenador": doc["_source"].get("coordenador"),
                "situacao": doc["_source"].get("situacao"),
                "versao": doc["_source"].get("versao"),
                "score": round(score_final, 2) * 100,  # Normalizando para porcentagem
                "score_nome": round(score_nome, 2) * 100,
                "score_resumo": round(score_resumo, 2) * 100
            })

        # Ordenar por score final
        cursos_final.sort(key=lambda x: x["score"], reverse=True)

        for curso in cursos_final[:15]:
            cursos_similares.append(    
                f"📌 Curso Similar: {curso['nome']}\n"
                f"📊 Similaridade: {curso['score']}%\n"
                f"👨‍🏫 Coordenador: {curso['coordenador']}\n"
                f"📌 Situação: {curso['situacao']}\n"
                f"🆕 Versão: {curso['versao']}\n"
                f"--------------------------------------------------\n"
            )

        if len(cursos_final) == 0:
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