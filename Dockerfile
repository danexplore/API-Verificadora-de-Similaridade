# Usa uma imagem oficial do Python
FROM python:3.11-slim

# Cria diretório da aplicação
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta da API (exemplo: 8000)
EXPOSE 8000

# Comando para iniciar a API (ajuste conforme seu app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
