#!/usr/bin/env python3
"""
Script para limpar dados de teste e importar dados reais do Elasticsearch
"""

from elasticsearch import Elasticsearch
import json
import csv
import time

# Conectar ao Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Indices
indices = ["cursos_reservas", "cursos_em_producao", "cursos_lancados", "cursos_inativos"]

def limpar_indices():
    """Remove todos os documentos dos índices"""
    print("🗑️  Limpando dados de teste...")
    for index in indices:
        try:
            if es.indices.exists(index=index):
                es.indices.delete(index=index)
                print(f"   ✅ Índice '{index}' deletado")
        except Exception as e:
            print(f"   ❌ Erro ao deletar '{index}': {e}")

def recriar_indices():
    """Recria os índices vazios"""
    print("\n📚 Recriando índices vazios...")
    
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
    
    for index in indices:
        try:
            es.indices.create(index=index, body=mapping)
            print(f"   ✅ Índice '{index}' recriado (vazio)")
        except Exception as e:
            print(f"   ❌ Erro ao criar '{index}': {e}")

def importar_json(arquivo_json, indice_destino="cursos_lancados"):
    """Importa dados de um arquivo JSON"""
    print(f"\n📥 Importando dados de {arquivo_json}...")
    try:
        with open(arquivo_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Se for uma lista de dicts
        if isinstance(dados, list):
            for i, item in enumerate(dados, 1):
                es.index(index=indice_destino, id=i, document=item)
            print(f"   ✅ {len(dados)} documentos importados para '{indice_destino}'")
        else:
            print(f"   ❌ Arquivo JSON deve conter uma lista de objetos")
    except FileNotFoundError:
        print(f"   ❌ Arquivo não encontrado: {arquivo_json}")
    except json.JSONDecodeError:
        print(f"   ❌ Erro ao decodificar JSON")
    except Exception as e:
        print(f"   ❌ Erro ao importar: {e}")

def importar_csv(arquivo_csv, indice_destino="cursos_lancados"):
    """Importa dados de um arquivo CSV"""
    print(f"\n📥 Importando dados de {arquivo_csv}...")
    try:
        with open(arquivo_csv, 'r', encoding='utf-8') as f:
            leitor = csv.DictReader(f)
            for i, linha in enumerate(leitor, 1):
                es.index(index=indice_destino, id=i, document=linha)
        print(f"   ✅ {i} documentos importados para '{indice_destino}'")
    except FileNotFoundError:
        print(f"   ❌ Arquivo não encontrado: {arquivo_csv}")
    except Exception as e:
        print(f"   ❌ Erro ao importar: {e}")

def importar_de_dicionario(dados, indice_destino="cursos_lancados"):
    """Importa dados de um dicionário Python"""
    print(f"\n📥 Importando dados para '{indice_destino}'...")
    try:
        if isinstance(dados, list):
            for i, item in enumerate(dados, 1):
                es.index(index=indice_destino, id=i, document=item)
            print(f"   ✅ {len(dados)} documentos importados")
        else:
            print(f"   ❌ Dados devem ser uma lista de dicts")
    except Exception as e:
        print(f"   ❌ Erro ao importar: {e}")

def listar_dados(indice="cursos_lancados", limite=10):
    """Lista documentos do índice"""
    try:
        resultado = es.search(index=indice, size=limite)
        total = resultado["hits"]["total"]["value"]
        print(f"\n📊 Total de documentos em '{indice}': {total}")
        
        if total > 0:
            print("\n📋 Primeiros documentos:")
            for hit in resultado["hits"]["hits"][:3]:
                print(f"   ID: {hit['_id']}")
                for chave, valor in hit["_source"].items():
                    print(f"      {chave}: {valor}")
                print()
    except Exception as e:
        print(f"❌ Erro ao listar: {e}")

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("GERENCIADOR DE DADOS - ELASTICSEARCH")
    print("=" * 60)
    
    # Opção 1: Limpar tudo
    if len(sys.argv) > 1 and sys.argv[1] == "limpar":
        limpar_indices()
        recriar_indices()
        print("\n✨ Índices limpos e recriados (vazios)")
        listar_dados()
    
    # Opção 2: Importar JSON
    elif len(sys.argv) > 2 and sys.argv[1] == "json":
        arquivo = sys.argv[2]
        indice = sys.argv[3] if len(sys.argv) > 3 else "cursos_lancados"
        importar_json(arquivo, indice)
        time.sleep(1)
        listar_dados(indice)
    
    # Opção 3: Importar CSV
    elif len(sys.argv) > 2 and sys.argv[1] == "csv":
        arquivo = sys.argv[2]
        indice = sys.argv[3] if len(sys.argv) > 3 else "cursos_lancados"
        importar_csv(arquivo, indice)
        time.sleep(1)
        listar_dados(indice)
    
    # Opção 4: Listar
    elif len(sys.argv) > 1 and sys.argv[1] == "listar":
        indice = sys.argv[2] if len(sys.argv) > 2 else "cursos_lancados"
        listar_dados(indice, limite=20)
    
    else:
        print("\n📖 MODO DE USO:\n")
        print("1. LIMPAR TODOS OS DADOS:")
        print("   python gerenciar_elasticsearch.py limpar\n")
        
        print("2. IMPORTAR DE JSON:")
        print("   python gerenciar_elasticsearch.py json arquivo.json [indice]")
        print("   Exemplo: python gerenciar_elasticsearch.py json cursos.json cursos_lancados\n")
        
        print("3. IMPORTAR DE CSV:")
        print("   python gerenciar_elasticsearch.py csv arquivo.csv [indice]")
        print("   Exemplo: python gerenciar_elasticsearch.py csv cursos.csv cursos_lancados\n")
        
        print("4. LISTAR DOCUMENTOS:")
        print("   python gerenciar_elasticsearch.py listar [indice]")
        print("   Exemplo: python gerenciar_elasticsearch.py listar cursos_lancados\n")
        
        print("=" * 60)
        print("📝 ESTRUTURA ESPERADA DOS DADOS:\n")
        print("JSON (arquivo.json):")
        print(json.dumps([
            {
                "nome": "Python Avançado",
                "resumo": "Conceitos avançados de Python",
                "coordenador": "João Silva",
                "situacao": "Ativo",
                "versao": "2.0"
            }
        ], ensure_ascii=False, indent=2))
        
        print("\nCSV (arquivo.csv):")
        print("nome,resumo,coordenador,situacao,versao")
        print("Python Avançado,Conceitos avançados de Python,João Silva,Ativo,2.0")
