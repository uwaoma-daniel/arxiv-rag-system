FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-hf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-hf.txt

RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY src/        ./src/
COPY api/        ./api/
COPY config.yaml .
COPY config-hf.yaml .
COPY entrypoint.py .

RUN mkdir -p logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "entrypoint.py"]
