FROM python:3.11-slim

WORKDIR /app

# Install system deps for bitsandbytes
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/ ./serve/
COPY config.yaml .

# Copy trained adapter if it exists (build after training)
COPY outputs/dpo_model/ ./outputs/dpo_model/

ENV BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
ENV ADAPTER_PATH="./outputs/dpo_model"

EXPOSE 8000

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
