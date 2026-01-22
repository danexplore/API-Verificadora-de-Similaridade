from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from upstash_redis import Redis
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel
import os
import requests
import unicodedata
import re
import json
import orjson
import secrets
import time

# =========================
# Configuração inicial
# =========================

if os.getenv("ENVIRONMENT") == "development":
    load_dotenv()

class ORJSONResponse(Response):
    media_type = "application/json"
    def render(self, content: any) -> bytes:
        return orjson.dumps(content)

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
    max_age=60 * 60 * 24
)

app.add_middleware(GZipMiddleware, minimum_size=500)

redis = Redis.from_env()

# =========================
# Utilidades
# =========================

def preparar_para_embedding(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = re.sub(r"[\[\]\(\)\:\-\_]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto

# =========================
# PIPEFY – OAUTH 2.0
# =========================

PIPEFY_API_URL = os.getenv("PIPEFY_API_URL", "https://api.pipefy.com/graphql")
PIPEFY_OAUTH_URL = os.getenv("PIPEFY_OAUTH_URL", "https://app.pipefy.com/oauth/token")
PIPEFY_CLIENT_ID = os.getenv("PIPEFY_CLIENT_ID")
PIPEFY_CLIENT_SECRET = os.getenv("PIPEFY_CLIENT_SECRET")

_pipefy_token_cache = None

def get_pipefy_token():
    response = requests.post(
        PIPEFY_OAUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": PIPEFY_CLIENT_ID,
            "client_secret": PIPEFY_CLIENT_SECRET
        },
        timeout=10
    )

    if response.status_code != 200:
        raise Exception(f"Erro OAuth Pipefy: {response.text}")

    data = response.json()
    return {
        "access_token": data["access_token"],
        "expires_at": time.time() + data["expires_in"] - 60
    }

def get_valid_pipefy_token():
    global _pipefy_token_cache

    if (
        _pipefy_token_cache is None
        or time.time() > _pipefy_token_cache["expires_at"]
    ):
        _pipefy_token_cache = get_pipefy_token()

    return _pipefy_token_cache["access_token"]

async def atualizar_pipefy(card_id, cursos_similares_str):
    try:
        token = get_valid_pipefy_token()

        mutation = f"""
        mutation {{
            updateCardField(input: {{
                card_id: "{card_id}",
                field_id: "cursos_similares",
                new_value: "{cursos_similares_str}"
            }}) {{
                card {{ id }}
            }}
        }}
        """

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            PIPEFY_API_URL,
            json={{"query": mutation}},
            headers=headers,
            timeout=10
        )

        if response.status_code != 200:
            print("[ERRO PIPEFY]", response.text)
            return "error"

        print("[PIPEFY] Campo atualizado com sucesso.")
        return "success"

    except Exception as e:
        print(f"[ERRO PIPEFY] {str(e)}")
        return "error"

# =========================
# Elasticsearch
# =========================

ELASTICSEARCH_URL = "https://daniel-elasticsearch.ekyhxs.easypanel.host"

client = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD")),
    max_retries=3,
    retry_on_timeout=True,
    request_timeout=10,
    connections_per_node=10
)

@lru_cache(maxsize=1)
def get_model():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    return SentenceTransformer("intfloat/e5-base-v2")

# =========================
# OpenAI
# =========================

async def avaliar_relevancia_ia(nome, resumo, cursos):
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if resumo == "":
        resumo = "Resumo do curso não fornecido."

    prompt = f"Curso principal:\nNome: {nome}\nResumo: {resumo}\n\nCursos similares:\n"
    for i, curso in enumerate(cursos):
        prompt += f"id: {i}\nnome: {curso['nome']}\n"

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "Avalie a similaridade entre cursos e responda em JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = client_openai.chat.completions.create(**payload)
    return json.loads(response.choices[0].message.content)

# =========================
# Segurança – Basic Auth
# =========================

USERS = {}
users_env = os.getenv("BASIC_AUTH_USERS")
if users_env:
    for pair in users_env.split(","):
        user, pwd = pair.split(":", 1)
        USERS[user.strip()] = pwd.strip()

security = HTTPBasic()

def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    password = USERS.get(credentials.username)
    if not password or not secrets.compare_digest(credentials.password, password):
        raise HTTPException(status_code=401, detail="Acesso negado.")
    return credentials

# =========================
# Endpoints
# =========================

@app.get("/")
async def root(credentials: HTTPBasicCredentials = Depends(verify_basic_auth)):
    return {"message": "API de Similaridade de Cursos Unyleya"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
