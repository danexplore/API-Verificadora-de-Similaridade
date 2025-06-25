from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from upstash_redis import Redis
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import unicodedata
import re
import json
from functools import lru_cache

if os.getenv("ENVIRONMENT") == "development":
    load_dotenv()

app = FastAPI(title="API de Similaridade de Cursos", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens
    allow_methods=["*"],
    allow_headers=["*"]
)

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

ELASTIC_URL_TOKEN = os.getenv("ELASTIC_URL_TOKEN")
# Configuração do Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL = f"https://daniel-elasticsearch.ekyhxs.easypanel.host"

# Inicializar cliente do Elasticsearch
client = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(os.getenv('ELASTIC_USERNAME'), os.getenv('ELASTIC_PASSWORD'))
)

@app.get("/")
async def root():
    return {"message": "API de Similaridade de Cursos Unyleya - Versão 1.0"}

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer('intfloat/e5-base-v2', cache_folder='/app/models')

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

        # Delete cursos with less than 3 stars:
        avaliacoes_dict = {k: v for k, v in avaliacoes_dict.items() if int(v["estrelas"]) >= 3}

        # Merge das informações da IA com os cursos
        for i, curso in enumerate(cursos_final, start=0):
            ia_data = avaliacoes_dict.get(str(i))
            if ia_data:
                curso["estrelas"] = int(ia_data["estrelas"])
                curso["comentario"] = ia_data["comentario"]
            else:
                curso["estrelas"] = 1
                curso["comentario"] = "Não avaliado pela IA."

        # Delete cursos with menos de 3 estrelas:
        cursos_final = [c for c in cursos_final if int(c["estrelas"]) >= 3]

        # Ordenar por estrelas (desc), depois por score
        cursos_final.sort(key=lambda x: (x.get("estrelas", 0), x["score"]), reverse=True)
 
        # Gerar string de cursos similares
        cursos_similares = ["🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n"]
        for curso in cursos_final:
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

        return cursos_similares_str
    
    except Exception as e:
        print(f"[ERRO] Erro ao processar IA ou atualizar Pipefy: {str(e)}")
        return {"message": "Erro ao processar: " + str(e)}

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

@app.get("/buscar/")
async def buscar_similaridade(
    nome: str,
    card_id: str = None,
    qtd_respostas: int = 50,
    resumo: str = None,
    situacao: str = None,
    versao: str = None,
    coordenador: str = None,
    background_tasks: BackgroundTasks = None,
    usar_ia: bool = True
):
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
    cache_key = f"buscar_similaridade:{nome}:{resumo}:{situacao}:{versao}:{coordenador}:{usar_ia}"
    cached_data = redis.get(cache_key)
    if cached_data:
        return {"message": "cache", "cursos_similares": cached_data}

    try:
        if not nome:
            raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

        nome_preparado = preparar_para_embedding(nome)
        nome_vector = get_model().encode(f'query: {nome_preparado}').tolist()

        resumo_preparado = preparar_para_embedding(resumo) if resumo else None
        resumo_vector = get_model().encode(f'passage: {resumo_preparado}').tolist() if resumo_preparado else None

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
        if usar_ia:
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
                                    "k": 150,
                                    "num_candidates": 300
                                }
                            }
                        }
                    },
                    "_source": ["nome", "coordenador", "situacao", "versao"]
                }
        else:
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
                                    "k": 150,
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
        scores_resumo = {r["_id"]: r["_score"] for r in res_resumo} if res_resumo else {}

        # Mesclar e calcular score final
        todos_ids = set(scores_nome.keys()).union(scores_resumo.keys())
        peso_nome = 0.7
        peso_resumo = 0.3

        cursos_final = []
        for _id in todos_ids:
            score_nome = scores_nome.get(_id, 0)
            score_resumo = scores_resumo.get(_id, 0)
            if not score_resumo == 0:
                score_final = (peso_nome * score_nome) + (peso_resumo * score_resumo)
            else:
                score_final = score_nome
            if score_final < 0.92:
                continue

            # Buscar o documento completo (de qualquer uma das buscas)
            doc = next((r for r in res_nome + res_resumo if r["_id"] == _id), None)
            if not doc:
                continue

            curso = {
                "nome": doc["_source"]["nome"],
                "coordenador": doc["_source"].get("coordenador"),
                "situacao": doc["_source"].get("situacao"),
                "versao": doc["_source"].get("versao"),
                "score": round(score_final, 2) * 100,  # Normalizando para porcentagem
                "score_nome": round(score_nome, 2) * 100
            }
            if score_resumo > 0:
                curso["score_resumo"] = round(score_resumo, 2) * 100
            else:
                curso["score_resumo"] = 0
            cursos_final.append(curso)

        # Ordenar por Elasticsearch
        cursos_final.sort(key=lambda x: x["score"], reverse=True)

        cursos_final = cursos_final[:qtd_respostas]

        if usar_ia:
            # Processar IA em background
            cursos_similares_str = await processar_ia(nome, resumo, cursos_final)
        else:
            cursos_similares_str = "🔍 Cursos Similares Encontrados:\n--------------------------------------------------\n" \
            "\n".join(
                f"📌 Curso Similar: {curso['nome']}\n"
                f"📊 Similaridade: {curso['score']}%\n"
                f"👨‍🏫 Coordenador: {curso['coordenador']}\n"
                f"📌 Situação: {curso['situacao']}\n"
                f"🆕 Versão: {curso['versao']}\n"
                f"--------------------------------------------------\n"
                for curso in cursos_final
            )

        redis.setex(cache_key, 600, cursos_similares_str)

        if card_id:
            response = background_tasks.add_task(
                atualizar_pipefy,
                card_id,
                cursos_similares_str
            )
            if response == "error":
                return {"message": "Erro ao atualizar o campo do cartão no Pipefy.", "cursos_similares": cursos_similares_str}
        return {"message": "Cursos similares encontrados.", "cursos_similares": cursos_similares_str}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar requisição: {str(e)}")
    
@app.get("/comparar-curso")
async def comparar_cursos_unicos(nome_principal: str, nome_similar: str, resumo_principal: str = ""):
    """
    Compara semanticamente um curso principal com um único curso similar.
    Retorna avaliação por estrelas e comentário explicativo da IA.
    """
    try:
        if not nome_principal or not nome_similar:
            raise HTTPException(status_code=400, detail="Nome do curso principal e do similar são obrigatórios.")

        # Preparar payload no mesmo formato usado na função de comparação múltipla
        curso = [{"nome": nome_similar}]

        avaliacoes = await avaliar_relevancia_ia(nome_principal, resumo_principal, curso)

        if not avaliacoes:
            return {"message": "A IA não conseguiu gerar uma avaliação."}

        avaliacao = avaliacoes

        return {
            "nome_similar": nome_similar,
            "estrelas": int(avaliacao["estrelas"]),
            "comentario": avaliacao["comentario"],
            "avaliacao_visual": "⭐" * int(avaliacao["estrelas"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao comparar cursos: {str(e)}")