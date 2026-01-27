#!/usr/bin/env python3
"""
Script para criar índices e dados de teste no Elasticsearch
"""

from elasticsearch import Elasticsearch
import time
import json

# Conectar ao Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Aguardar Elasticsearch estar pronto
print("⏳ Aguardando Elasticsearch ficar pronto...")
for i in range(30):
    try:
        es.info()
        print("✅ Elasticsearch conectado!")
        break
    except Exception as e:
        if i < 29:
            print(f"   Tentativa {i+1}/30... Aguarde...")
            time.sleep(2)
        else:
            print(f"❌ Não foi possível conectar ao Elasticsearch: {e}")
            exit(1)

# Definir mapeamento padrão para todos os índices
mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "nome": {"type": "text"},
            "resumo": {"type": "text"},
            "coordenador": {"type": "keyword"},
            "situacao": {"type": "keyword"},
            "versao": {"type": "keyword"}
        }
    }
}

# Índices a criar
indices = ["cursos_reservas", "cursos_em_producao", "cursos_lancados", "cursos_inativos"]

# Criar índices
print("\n📚 Criando índices...")
for index in indices:
    try:
        if es.indices.exists(index=index):
            print(f"   ⚠️  Índice '{index}' já existe. Pulando...")
        else:
            es.indices.create(index=index, body=mapping)
            print(f"   ✅ Índice '{index}' criado!")
    except Exception as e:
        print(f"   ❌ Erro ao criar índice '{index}': {e}")

# Dados de teste
cursos_teste = [
    {
        "nome": "Python Básico",
        "resumo": "Aprenda os fundamentos de Python, variáveis, loops, funções e estruturas de dados",
        "coordenador": "João Silva",
        "situacao": "Ativo",
        "versao": "1.0"
    },
    {
        "nome": "Python Avançado",
        "resumo": "Conceitos avançados de Python, programação orientada a objetos, decoradores e async",
        "coordenador": "Maria Santos",
        "situacao": "Ativo",
        "versao": "2.0"
    },
    {
        "nome": "JavaScript Básico",
        "resumo": "Introdução ao JavaScript, DOM, eventos e manipulação de página web",
        "coordenador": "Pedro Costa",
        "situacao": "Ativo",
        "versao": "1.5"
    },
    {
        "nome": "JavaScript Avançado",
        "resumo": "Conceitos avançados de JavaScript, ES6+, async await, promises e design patterns",
        "coordenador": "Ana Ferreira",
        "situacao": "Ativo",
        "versao": "2.1"
    },
    {
        "nome": "Desenvolvimento Web com React",
        "resumo": "Aprenda React, componentes, hooks, state management e routing para criar aplicações modernas",
        "coordenador": "Carlos Mendes",
        "situacao": "Ativo",
        "versao": "1.0"
    },
    {
        "nome": "Node.js e Express",
        "resumo": "Construção de APIs com Node.js e Express, middleware, autenticação e conexão com banco de dados",
        "coordenador": "João Silva",
        "situacao": "Ativo",
        "versao": "1.2"
    },
    {
        "nome": "Banco de Dados SQL",
        "resumo": "Fundamentos de SQL, queries, joins, índices e otimização de consultas",
        "coordenador": "Lucia Ribeiro",
        "situacao": "Ativo",
        "versao": "1.0"
    },
    {
        "nome": "MongoDB e NoSQL",
        "resumo": "Banco de dados MongoDB, BSON, queries, agregação e modelagem de dados",
        "coordenador": "Roberto Dias",
        "situacao": "Ativo",
        "versao": "1.1"
    },
    {
        "nome": "Docker e Containerização",
        "resumo": "Aprenda Docker, criar imagens, containers, docker-compose e orquestração",
        "coordenador": "Fernanda Lima",
        "situacao": "Ativo",
        "versao": "1.3"
    },
    {
        "nome": "Git e Controle de Versão",
        "resumo": "Controle de versão com Git, branches, merge, rebase e fluxo de trabalho colaborativo",
        "coordenador": "Gustavo Martins",
        "situacao": "Ativo",
        "versao": "1.0"
    }
]

# Inserir dados de teste
print("\n📝 Inserindo dados de teste...")
try:
    for i, curso in enumerate(cursos_teste):
        es.index(
            index="cursos_lancados",
            id=i+1,
            document=curso
        )
    print(f"   ✅ {len(cursos_teste)} cursos inseridos em 'cursos_lancados'")
except Exception as e:
    print(f"   ❌ Erro ao inserir dados: {e}")

# Inserir alguns em outro índice também
print("\n📝 Inserindo dados de teste em outros índices...")
try:
    for i, curso in enumerate(cursos_teste[:3]):
        es.index(
            index="cursos_em_producao",
            id=i+1,
            document=curso
        )
    print(f"   ✅ 3 cursos inseridos em 'cursos_em_producao'")
except Exception as e:
    print(f"   ❌ Erro ao inserir dados: {e}")

# Aguardar indexação
print("\n⏳ Aguardando indexação...")
time.sleep(2)

# Verificar dados
print("\n🔍 Verificando dados...")
try:
    result = es.search(index="cursos_lancados", size=100)
    total = result["hits"]["total"]["value"]
    print(f"   ✅ Total de cursos em 'cursos_lancados': {total}")
except Exception as e:
    print(f"   ❌ Erro ao verificar: {e}")

print("\n✨ Setup concluído! Você já pode testar a API.")
print("\n📋 Próximos passos:")
print("   1. Execute a API: python -c \"import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=10000)\"")
print("   2. Teste em outro terminal com:")
print("      curl.exe -X POST http://127.0.0.1:10000/buscar -H \"Content-Type: application/json\" -H \"Authorization: Basic YWRtaW46MTIzNDU2Nzg5\" --data-binary \"@test_buscar.json\"")
