import pandas as pd
from upstash_redis import Redis
import os, time
from dotenv import load_dotenv

load_dotenv()
redis = Redis(url=os.getenv("UPSTASH_REDIS_URL"))

df = pd.read_excel("cursos-elastic-search.xlsx")
df = df.where(pd.notnull(df), None)

for _, row in df.iterrows():
    redis.hset(f"curso:{row['ID']}", mapping={
        "nome": row["Nome do Curso"],
        "coordenador": row["Coordenador Titular"],
        "situacao": row["Evolução Acadêmica"],
        "versao": row["Versão do Curso"],
        "segmento": row["Segmento"],
        "pre_matriculas_2024": row.get("Pré-Matrículas 2024", 0),
        "pre_matriculas_2025": row.get("Pré-Matrículas 2025", 0),
        "matriculas_2024": row.get("Matrículas 2024", 0),
        "matriculas_2025": row.get("Matrículas 2025", 0),
        "data_atualizacao": time.strftime("%Y-%m-%d")
    })
print("Migração de meta-dados para o Redis concluída.") 