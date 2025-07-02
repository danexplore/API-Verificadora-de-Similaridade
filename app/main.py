from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from upstash_redis import Redis
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import unicodedata
import re
import json
from functools import lru_cache
import orjson
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends
import secrets

class ORJSONResponse(Response):
    media_type = "application/json"
    def render(self, content: any) -> bytes:
        return orjson.dumps(content)

if os.getenv("ENVIRONMENT") == "development":
    load_dotenv()

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
            "Você é um especialista em análise educacional. Com base no nome e resumo (se fornecido) do curso principal, "
            "avalie semanticamente a similaridade com o curso listado. Considere que o curso pode ter diferenças de enfoque, "
            "mas ainda assim pode ser relevante. Retorne:\n"
            "- Uma nota de 1 a 5 estrelas (apenas número inteiro)\n"
            "- Um comentário explicativo justificando a nota com um parágrafo\n\n"
            "IMPORTANTE: sua resposta deve estar no formato JSON, sem texto adicional. Exemplo:\n"
            '{"id": "1", "estrelas": 4, "comentario": "Tem grande relação temática, porém o enfoque é diferente."}'
        )
    else:
        instrucoes = (
            "Avalie a relevância de cursos em relação a um curso principal com base em diferenças e semelhanças.\n\n"
            "Foque nas diferenças práticas e teóricas entre os cursos listados e o curso principal, mesmo em casos de semelhança. "
            "Avalie com uma nota de 1 a 5 estrelas, onde apenas números inteiros são usados.\n\n"
            "# Instruções\n\n"
            "- Para cada curso listado, avalie a relevância em relação ao curso principal usando uma nota de 1 a 5 estrelas. "
            "Use apenas números inteiros.\n"
            "- Ao fornecer um comentário, foque em como os cursos se diferenciam um do outro, além de suas semelhanças. "
            "Se forem muito similares, destaque as diferenças práticas e teóricas que justificariam a oferta de ambos, ou se um curso pode sobrepor o outro.\n"
            "- Se um curso receber menos de 3 estrelas, o campo de comentário deve permanecer vazio.\n\n"
            "# Output Format\n\n"
            "A saída deve estar no formato de lista JSON sem texto adicional fora desse formato.\n\n"
            "# Examples\n\n"
            "## Example Input:\n\n"
            "- Curso principal: [Nome e resumo do curso principal]\n"
            "- Cursos listados: \n"
            "  1. Curso A: [Nome e resumo do curso A]\n"
            "  2. Curso B: [Nome e resumo do curso B]\n\n"
            "## Exemplo de Saída :\n\n"
            "[\n"
            "  {\"id\": \"1\", \"estrelas\": \"4\", \"comentario\": \"Embora ambos abordem o mesmo tema, este curso se concentra em aplicações práticas, enquanto o curso principal é mais teórico.\"},\n"
            "  {\"id\": \"2\", \"estrelas\": \"3\", \"comentario\": \"Os cursos possuem similaridade temática, mas este foca mais em uma abordagem diferente de ensino.\"},\n"
            "  {\"id\": \"3\", \"estrelas\": \"2\", \"comentario\": \"\"}\n"
            "]\n\n"
            "# Notas\n\n"
            "- Avalie como os cursos podem ser diferentes um do outro e justifique esses pontos.\n"
            "- Mantenha os comentários claros e específicos, indicando diferenças práticas, abordagens, ou focos de estudo.\n"
            "- Utilize a escala de estrelas para ajudar a distinguir cursos que podem parecer similares, mas têm diferenças significativas a serem consideradas."
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

def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    password = USERS.get(credentials.username)
    if not password or not secrets.compare_digest(credentials.password, password):
        raise HTTPException(status_code=401, detail="Acesso negado.", headers={"WWW-Authenticate": "Basic"})
    return credentials

@app.get("/")
async def root(credentials: HTTPBasicCredentials = Depends(verify_basic_auth)):
    return {"message": "API de Similaridade de Cursos Unyleya - Versão 1.0"}

@app.get("/buscar")
async def buscar_similaridade(
    nome: str,
    card_id: str = None,
    qtd_respostas: int = 50,
    resumo: str = None,
    situacao: str = None,
    versao: str = None,
    coordenador: str = None,
    background_tasks: BackgroundTasks = None,
    usar_ia: bool = True,
    credentials: HTTPBasicCredentials = Depends(verify_basic_auth)
):
    """
    Busca cursos similares usando Upstash Vector e recupera meta-dados do Redis.
    """
    cache_key = f"buscar_similaridade:{nome}:{resumo}:{situacao}:{versao}:{coordenador}:{usar_ia}"
    cached_data = redis.json.get(cache_key)
    if cached_data:
        return cached_data[0]

    try:
        if not nome:
            raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

        model = get_model()
        nome_preparado = preparar_para_embedding(nome)
        vector_nome = model.encode(f"query: {nome_preparado}").tolist()

        VECTOR_SEARCH_URL = os.getenv("UPSTASH_VECTOR_URL") + "/query"
        VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")
        headers = {
            "Authorization": f"Bearer {VECTOR_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "vector": vector_nome,
            "topK": qtd_respostas
        }
        response = requests.post(VECTOR_SEARCH_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Erro na busca vetorial: {response.text}")
        resultados = response.json()

        cursos_final = []
        for res in resultados.get("matches", []):
            curso_meta = redis.hgetall(f"curso:{res['id']}")
            if not curso_meta:
                continue

            # Filtros opcionais
            if situacao and curso_meta.get("situacao") not in situacao.split(","):
                continue
            if versao and curso_meta.get("versao") not in versao.split(","):
                continue
            if coordenador and coordenador.lower() not in (curso_meta.get("coordenador") or '').lower():
                continue

            curso_meta["score"] = round(res["score"], 2) * 100
            cursos_final.append(curso_meta)

        cursos_final.sort(key=lambda x: x["score"], reverse=True)
        cursos_final = cursos_final[:qtd_respostas]

        if usar_ia:
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

        redis.json.set(cache_key, path="$", value=result, nx=True)

        if card_id:
            response = background_tasks.add_task(
                atualizar_pipefy,
                card_id,
                cursos_similares_str
            )
            if response == "error":
                return {"message": "Erro ao atualizar o campo do cartão no Pipefy.", "cursos_similares": cursos_similares_str}
        return result

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