from fastapi import FastAPI, HTTPException, BackgroundTasks
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import requests
import unicodedata
import re
import json

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

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"  # substitua se for outro
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # salve no seu .env

# Criar sessão para requests do DeepSeek
deepseek_session = requests.Session()
deepseek_session.headers.update({
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Content-Type": "application/json"
})

# Configuração do Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL = f"https://daniel-elasticsearch.ekyhxs.easypanel.host"

# Inicializar cliente do Elasticsearch
client = Elasticsearch(
    ELASTICSEARCH_URL,
    basic_auth=(os.getenv('ELASTIC_USERNAME'), os.getenv('ELASTIC_PASSWORD'))
)


# Inicializar modelo de embeddings
model = SentenceTransformer('intfloat/e5-base-v2', cache_folder='/app/models')

# Inicializar FastAPI
app = FastAPI(title="API de Similaridade de Cursos", version="1.0")

def avaliar_relevancia_ia(nome, resumo, cursos):
    import re

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
            "Você é um especialista em análise educacional. Com base no nome e resumo (se fornecido) do curso principal, "
            "avalie semanticamente a similaridade com os cursos listados. Considere que os cursos podem ter diferenças de enfoque, "
            "mas ainda assim podem ser relevantes.\n Com base nisso retorne:\n"
            "-- Uma nota de 1 a 5 estrelas para o curso fornecido segundo sua relevancia (apenas número inteiro)\n"
            "-- Um comentário breve explicando a diferença.\n\n"
            "IMPORTANTE: Não gere comentarios para cursos abaixo de 3 estrelas, deixe em branco como no exemplo a seguir.\n"
            "IMPORTANTE: sua resposta deve estar no formato JSON, sem texto adicional. Exemplo:\n"
            '[{"id": "1", "estrelas": "4", "comentario": "Tem grande relação temática, porém o enfoque é diferente."},'
            '{"id": "2", "estrelas": "1", "comentario": ""}, ...]'
        )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": instrucoes},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = deepseek_session.post(DEEPSEEK_URL, json=payload)
        response.raise_for_status()
        conteudo = response.json()["choices"][0]["message"]["content"]

        # 🔧 Corrigir conteúdo com marcação Markdown tipo ```json ... ```
        if conteudo.strip().startswith("```json"):
            conteudo = re.sub(r"^```json\s*|\s*```$", "", conteudo.strip(), flags=re.DOTALL)

        # Validar se o conteúdo é um JSON válido
        try:
            resultado = json.loads(conteudo)
            print(resultado)
            if isinstance(resultado, list) or isinstance(resultado, dict):
                return resultado
            else:
                raise ValueError("O retorno não é um JSON válido.", conteudo)
        except json.JSONDecodeError as e:
            print("[ERRO IA] Conteúdo não é JSON válido:", e)
            return []

    except Exception as e:
        print(f"[ERRO IA] {e}")
        return []


@app.get("/")
async def home():
    return {"message": "API de Similaridade de Cursos Online!"}

# Função para rodar a chamada à IA e atualizar o Pipefy em background
def processar_ia(nome, resumo, cursos_final):
    try:
        # Avaliar relevância com IA
        avaliacoes_ia = avaliar_relevancia_ia(nome, resumo or "", cursos_final)
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

def atualizar_pipefy(card_id, cursos_similares_str):
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
    qtd_respostas: int = 7,
    resumo: str = None,
    situacao: str = None,
    versao: str = None,
    coordenador: str = None,
    background_tasks: BackgroundTasks = None,
    usar_ia: bool = True
):
    """
    Busca cursos similares no Elasticsearch usando nome e resumo do curso com busca híbrida (texto + vetor).
    """
    try:
        if not nome:
            raise HTTPException(status_code=400, detail="Nome do curso é obrigatório.")

        nome_preparado = preparar_para_embedding(nome)
        nome_vector = model.encode(f'query: {nome_preparado}').tolist()

        resumo_preparado = preparar_para_embedding(resumo) if resumo else None
        resumo_vector = model.encode(f'passage: {resumo_preparado}').tolist() if resumo_preparado else None

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
                    "size": 25,
                    "query": {
                        "bool": {
                            "filter": filters,
                            "must": {
                                "knn": {
                                    "field": vector_field,
                                    "query_vector": vector,
                                    "k": 75,
                                    "num_candidates": 250
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
            cursos_similares_str = processar_ia(nome, resumo, cursos_final)
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
        
        if card_id:
            response = background_tasks.add_task(
                atualizar_pipefy,
                card_id,
                cursos_similares_str
            )
            if response == "error":
                return {"message": "Erro ao atualizar o campo do cartão no Pipefy.", "cursos_similares": cursos_similares_str}
        return {"message": "Campo do cartão atualizado com sucesso.", "cursos_similares": cursos_similares_str}


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

        avaliacoes = avaliar_relevancia_ia(nome_principal, resumo_principal, curso)

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