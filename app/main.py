from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from typing import Optional
from upstash_redis import Redis
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import unicodedata
import re
import json
from functools import lru_cache
import orjson
from fastapi.security import HTTPBearer, HTTPBasic, HTTPBasicCredentials, HTTPAuthorizationCredentials
from fastapi import Depends
from fastapi import Request
import secrets
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class ORJSONResponse(Response):
    media_type = "application/json"
    def render(self, content: any) -> bytes:
        return orjson.dumps(content)

load_dotenv()

if os.getenv("ENVIRONMENT") == "development":
    pass

app = FastAPI(
    title="API de Similaridade de Cursos",
    version="1.0",
    default_response_class=ORJSONResponse
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=60*60*24
)
app.add_middleware(GZipMiddleware, minimum_size=500)

redis = Redis.from_env()

# Inicializar cliente Elasticsearch
elasticsearch_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
elasticsearch_api_key = os.getenv("ELASTICSEARCH_API_KEY", "")

try:
    if elasticsearch_api_key:
        es_client = Elasticsearch([elasticsearch_url], api_key=elasticsearch_api_key)
    else:
        es_client = Elasticsearch([elasticsearch_url])
    # Testar conexão
    es_client.info()
    print("[OK] Elasticsearch conectado com sucesso")
except Exception as e:
    print(f"[WARNING] Nao foi possivel conectar ao Elasticsearch: {e}")
    es_client = None

def preparar_para_embedding(texto: str) -> str:
    # Remover acentos
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    # Remover símbolos que não ajudam semanticamente
    texto = re.sub(r"[\[\]\(\)\:\-\_]", " ", texto)
    # Remover múltiplos espaços e deixar minúsculo
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto

# Configuração do Pipefy
PIPEFY_API_URL = "https://api.pipefy.com/graphql"
PIPEFY_API_TOKEN = os.getenv('PIPEFY_API_TOKEN')

@lru_cache(maxsize=1)
def get_model():
    # Optionally set threadpool limit for performance
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    return SentenceTransformer('intfloat/e5-base-v2')

async def avaliar_relevancia_ia(nome, resumo, cursos):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")
    client_openai = OpenAI(api_key=openai_api_key)

    if resumo == "":
        resumo = "Resumo do curso não fornecido, continue a análise somente com o nome do curso."

    prompt = (
        f"Curso principal:\n Nome: {nome}\nResumo: {resumo}\n\n"
        f"Cursos similares:\n"
    )
    for i, curso in enumerate(cursos, start=0):
        prompt += f"id: {i}\nnome: {curso['nome']}\n"

    if len(cursos) == 1:
        instrucoes = (
            "Você é um especialista em educação. Avalie a similaridade entre o curso principal e o curso listado de forma OBJETIVA.\n\n"
            "ESCALA:\n"
            "5 = Praticamente idênticos\n"
            "4 = Muito similares\n"
            "3 = Moderadamente similares\n"
            "2 = Pouca similaridade\n"
            "1 = Nenhuma relação\n\n"
            "Retorne APENAS JSON sem texto adicional:\n"
            '{"id": "0", "estrelas": 4, "comentario": "Ambos cobrem o mesmo tema"}'
        )
    else:
        instrucoes = (
            "Você é um especialista em educação. Avalie cada curso comparando com o principal de forma OBJETIVA.\n\n"
            "ESCALA:\n"
            "5 = Praticamente idênticos\n"
            "4 = Muito similares\n"
            "3 = Moderadamente similares\n"
            "2 = Pouca similaridade\n"
            "1 = Nenhuma relação\n\n"
            "REGRAS:\n"
            "- Comentário OBRIGATÓRIO apenas se estrelas >= 3\n"
            "- Comentário BREVE (máximo 10 palavras)\n"
            "- Indique: diferenças de enfoque, público-alvo ou nível\n\n"
            "Retorne APENAS JSON (lista), sem texto adicional:\n"
            "[\n"
            "  {\"id\": \"0\", \"estrelas\": 4, \"comentario\": \"Mesmo tema, enfoque diferente\"},\n"
            "  {\"id\": \"1\", \"estrelas\": 2, \"comentario\": \"\"}\n"
            "]"
        )

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "system",
                "content": instrucoes
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5
    }
    try:
        response = client_openai.chat.completions.create(**payload)

        resposta_ia_str = response.choices[0].message.content
        if not resposta_ia_str:
            raise ValueError("Resposta da IA está vazia.")
        conteudo = json.loads(resposta_ia_str)
        # Validar se o conteúdo é um JSON válido
        if isinstance(conteudo, list) or isinstance(conteudo, dict):
            return conteudo
        else:
            raise ValueError("O retorno não é um JSON válido.", conteudo)
    except Exception as e:
        print(f"[ERRO IA] {e}")
        return []

async def processar_ia(nome, resumo, cursos_final):
    try:
        # Avaliar relevância com IA
        avaliacoes_ia = await avaliar_relevancia_ia(nome, resumo or "", cursos_final)
        avaliacoes_dict = {item["id"]: item for item in avaliacoes_ia}

        # Merge das informações da IA com os cursos
        for i, curso in enumerate(cursos_final, start=0):
            ia_data = avaliacoes_dict.get(str(i))
            if ia_data:
                curso["estrelas"] = int(ia_data["estrelas"])
                curso["comentario"] = ia_data["comentario"]
            else:
                curso["estrelas"] = 1
                curso["comentario"] = "Não avaliado pela IA."

        # Filtrar cursos com menos de 3 estrelas
        cursos_filtrados = [c for c in cursos_final if int(c["estrelas"]) >= 3]

        # Ordenar por estrelas (desc), depois por score
        cursos_filtrados.sort(key=lambda x: (x.get("estrelas", 0), x["score"]), reverse=True)
 
        # Gerar string de cursos similares
        cursos_similares = ["🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n"]
        for curso in cursos_filtrados:
            cursos_similares.append(
                f"📌 Curso Similar: {curso['nome']}\n"
                f"📊 Similaridade: {curso['score']}%\n"
                f"👨‍🏫 Coordenador: {curso['coordenador']}\n"
                f"📌 Situação: {curso['situacao']}\n"
                f"🆕 Versão: {curso['versao']}\n"
                f"🌟 Avaliação IA: {'⭐' * curso['estrelas']}\n"
                f"🧠 Comentário: {curso['comentario']}\n"
                f"--------------------------------------------------\n"
            )
        cursos_similares_str = "\n".join(cursos_similares)

        return cursos_similares_str, cursos_filtrados
    
    except Exception as e:
        print(f"[ERRO] Erro ao processar IA ou atualizar Pipefy: {str(e)}")
        return {"message": "Erro ao processar: " + str(e)}, []

async def atualizar_pipefy(card_id, cursos_similares_str):
    try:
        # Atualizar no Pipefy
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
            print("[ERRO] Falha ao atualizar o campo do cartão no Pipefy.")
            return "error"
        else:
            print("[SUCESSO] Campo do cartão atualizado no Pipefy.")
        
        return "success"
    except Exception as e:
        print(f"[ERRO] Erro ao atualizar Pipefy: {str(e)}")
        return "error"

# Parse users from env
USERS = {}
users_env = os.getenv("BASIC_AUTH_USERS")
if users_env:
    for pair in users_env.split(","):
        if ":" in pair:
            user, pwd = pair.split(":", 1)
            USERS[user.strip()] = pwd.strip()

security = HTTPBasic()

basic = HTTPBasic(auto_error=False)
bearer = HTTPBearer(auto_error=False)

API_TOKEN = os.getenv("API_TOKEN")  # defina no Render

def auth_mixed(
    request: Request,
    basic_credentials: HTTPBasicCredentials = Depends(basic),
    bearer_credentials: HTTPAuthorizationCredentials = Depends(bearer)
):
    # 1️⃣ Tenta Bearer (Pipefy)
    if bearer_credentials:
        if bearer_credentials.credentials == API_TOKEN:
            return True

    # 2️⃣ Fallback para Basic Auth (humano / curl)
    if basic_credentials:
        user = basic_credentials.username
        pwd = basic_credentials.password
        if USERS.get(user) == pwd:
            return True

    raise HTTPException(
        status_code=401,
        detail="Não autenticado"
    )


def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    password = USERS.get(credentials.username)
    if not password or not secrets.compare_digest(credentials.password, password):
        raise HTTPException(status_code=401, detail="Acesso negado.", headers={"WWW-Authenticate": "Basic"})
    return credentials

@app.get("/")
async def root(credentials: HTTPBasicCredentials = Depends(verify_basic_auth)):
    return {"message": "API de Similaridade de Cursos Unyleya - Versão 1.0"}

class CourseSimilaritySearch(BaseModel):
    nome: str
    card_id: Optional[str] = None
    qtd_respostas: int = 50
    resumo: Optional[str] = None
    situacao: Optional[str] = None
    versao: Optional[str] = None
    coordenador: Optional[str] = None
    usar_ia: bool = True

@app.post("/buscar")
async def buscar_similaridade(payload: CourseSimilaritySearch, credentials: HTTPBasicCredentials = Depends(verify_basic_auth), background_tasks: BackgroundTasks = None):
    """
    Busca cursos similares no Elasticsearch usando nome e resumo do curso com busca híbrida (texto + vetor).

    Args:
        nome (str): Nome do curso a ser buscado.
        card_id (str, optional): ID do cartão no Pipefy para atualizar com os resultados. Default é None.
        qtd_respostas (int, optional): Quantidade de respostas a serem retornadas. Default é 50.
        resumo (str, optional): Resumo do curso a ser buscado. Default é None.
        situacao (str, optional): Situação do curso para filtro. Default é None.
        versao (str, optional): Versão do curso para filtro. Default é None.
        coordenador (str, optional): Coordenador do curso para filtro. Default é None.
        background_tasks (BackgroundTasks, optional): Tarefas em segundo plano para atualizar Pipefy. Default é None.
        usar_ia (bool, optional): Se True, processa IA para avaliar relevância dos cursos encontrados. Default é True.
    
    Returns:
        dict: Dicionário com os cursos similares encontrados e suas informações, e caso possua o card_id vai atualizar o campo no Pipefy com os resultados.
    """
    nome = payload.nome.strip()
    card_id = payload.card_id.strip() if payload.card_id else None
    qtd_respostas = payload.qtd_respostas
    situacao = payload.situacao.strip() if payload.situacao else None
    versao = payload.versao.strip() if payload.versao else None
    coordenador = payload.coordenador.strip() if payload.coordenador else None
    usar_ia = payload.usar_ia
    resumo = payload.resumo.strip() if payload.resumo else None

    cache_key = f"buscar_similaridade:{nome}:{resumo}:{situacao}:{versao}:{coordenador}:{usar_ia}"
    try:
        cached_data = redis.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        print(f"Aviso: Erro ao recuperar cache: {e}")

    try:
        if not nome:
            raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

        if not es_client:
            raise HTTPException(
                status_code=503, 
                detail="Serviço de busca de cursos indisponível. Elasticsearch não está disponível."
            )

        # Obter modelo para embeddings
        model = get_model()
        
        # Preparar textos para embedding
        nome_preparado = preparar_para_embedding(nome)
        resumo_preparado = preparar_para_embedding(resumo) if resumo else ""
        
        # Gerar embeddings
        texto_embedding = nome if not resumo else f"{nome} {resumo}"
        embedding = model.encode(texto_embedding).tolist()
        
        # Preparar filtros
        filters = []
        if situacao:
            filters.append({"term": {"situacao.keyword": situacao}})
        if versao:
            filters.append({"term": {"versao.keyword": versao}})
        if coordenador:
            filters.append({"term": {"coordenador.keyword": coordenador}})
        
        # Construir query para Elasticsearch
        query = {
            "bool": {
                "should": [
                    {
                        "match": {
                            "nome": {
                                "query": nome_preparado,
                                "boost": 2
                            }
                        }
                    },
                    {
                        "match": {
                            "resumo": {
                                "query": resumo_preparado if resumo_preparado else nome_preparado,
                                "boost": 1
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
        
        if filters:
            query["bool"]["filter"] = filters
        
        # Buscar em todos os 4 índices
        indices = ["cursos_reservas", "cursos_em_producao", "cursos_lancados", "cursos_inativos"]
        
        cursos_final = []
        
        for index in indices:
            try:
                response = es_client.search(
                    index=index,
                    body={
                        "query": query,
                        "size": qtd_respostas,
                        "min_score": 0.1
                    }
                )
                
                for hit in response.get("hits", {}).get("hits", []):
                    doc = hit["_source"]
                    score = hit.get("_score", 0)
                    
                    # Normalizar score do Elasticsearch para porcentagem (0-100)
                    # Score máximo do ES é geralmente 20-30, então dividimos por 30 e multiplicamos por 100
                    # Isso dá uma distribuição mais realista
                    normalized_score = min(100, max(0, (score / 30) * 100))
                    
                    curso = {
                        "nome": doc.get("nome"),
                        "coordenador": doc.get("coordenador"),
                        "situacao": doc.get("situacao"),
                        "versao": doc.get("versao"),
                        "score": round(normalized_score, 1),  # Score de 0 a 100
                        "score_nome": round(normalized_score, 1)
                    }
                    cursos_final.append(curso)
            except Exception as e:
                print(f"Erro ao buscar no índice {index}: {e}")
                continue
        
        # Remover duplicatas baseado no nome
        nomes_vistos = set()
        cursos_unicos = []
        for curso in cursos_final:
            if curso["nome"] not in nomes_vistos:
                nomes_vistos.add(curso["nome"])
                cursos_unicos.append(curso)
        
        cursos_final = cursos_unicos
        
        # Ordenar por score (Elasticsearch)
        cursos_final.sort(key=lambda x: x["score"], reverse=True)

        cursos_final = cursos_final[:qtd_respostas]

        if usar_ia:
            # Processar IA em background
            cursos_similares_str, cursos_filtrados = await processar_ia(nome, resumo, cursos_final)
        else:
            cursos_similares_str = "🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n" \
            + "\n".join(
                f"📌 Curso Similar: {curso['nome']}\n"
                f"📊 Similaridade: {curso['score']}%\n"
                f"👨‍🏫 Coordenador: {curso['coordenador']}\n"
                f"📌 Situação: {curso['situacao']}\n"
                f"🆕 Versão: {curso['versao']}\n"
                f"--------------------------------------------------\n"
                for curso in cursos_final
            )
            cursos_filtrados = cursos_final
        result = {
            "message": "Cursos similares encontrados.",
            "nome": nome,
            "cursos_similares": cursos_similares_str,
            "cursos_similares_json": cursos_filtrados,
            "qtd_cursos_encontrados": len(cursos_filtrados)
        }

        try:
            redis.setex(cache_key, 3600, json.dumps(result))
        except Exception as e:
            print(f"Aviso: Erro ao salvar cache: {e}")

        if card_id:
            background_tasks.add_task(
                atualizar_pipefy,
                card_id,
                cursos_similares_str
            )
        return result


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {str(e)}")

@app.get("/comparar-curso")
async def comparar_cursos_unicos(
    nome_principal: str,
    nome_similar: str,
    resumo_principal: str = "",
    credentials: HTTPBasicCredentials = Depends(verify_basic_auth)
):
    """
    Compara semanticamente um curso principal com um único curso similar.
    Retorna avaliação por estrelas e comentário explicativo da IA.
    """
    redis_key = f"comparar_cursos_unicos:{nome_principal}:{nome_similar}:{resumo_principal}"
    cached_data = redis.get(redis_key)
    if cached_data:
        return json.loads(cached_data)
    try:
        if not nome_principal or not nome_similar:
            raise HTTPException(status_code=400, detail="Nome do curso principal e do similar são obrigatórios.")

        # Preparar payload no mesmo formato usado na função de comparação múltipla
        curso = [{"nome": nome_similar}]

        avaliacoes = await avaliar_relevancia_ia(nome_principal, resumo_principal, curso)

        if not avaliacoes:
            return {"message": "A IA não conseguiu gerar uma avaliação."}

        avaliacao = avaliacoes

        curso_similar = {
            "nome_similar": nome_similar,
            "estrelas": int(avaliacao["estrelas"]),
            "comentario": avaliacao["comentario"],
            "avaliacao_visual": "⭐" * int(avaliacao["estrelas"])
        }
        redis.setex(redis_key, 600, json.dumps(curso_similar))
        return json.dumps(curso_similar)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao comparar cursos: {str(e)}")
    

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/refresh")
async def refresh_cache():
    redis.flushdb()
    return {"message": "Cache refreshed successfully."}

