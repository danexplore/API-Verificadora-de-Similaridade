FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV USE_TORCH=1
ENV USE_ORT_MODEL=0
ENV USE_QUANTIZED_MODEL=0

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
