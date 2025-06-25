# Stage 1: build environment
FROM python:3.10.7-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1
ENV USE_ORT_MODEL=0
ENV USE_QUANTIZED_MODEL=0

WORKDIR /app

# Instala dependências de build
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl

COPY requirements.txt .

# Instala pacotes em uma pasta separada
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime (imagem final, menor e limpa)
FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1
ENV USE_ORT_MODEL=0
ENV USE_QUANTIZED_MODEL=0

WORKDIR /app

# Copia pacotes já instalados da imagem anterior
COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]