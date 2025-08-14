FROM python:3.11-slim

WORKDIR /app

# System deps (+ supervisord)
RUN apt-get update && apt-get install -y \
    build-essential python3-dev supervisor && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY src ./src
COPY ui ./ui
COPY data ./data
COPY supervisord.conf .
COPY README.md .
ENV PYTHONPATH=/app

EXPOSE 8501 8000

# Run Streamlit + FastAPI together
CMD ["supervisord", "-c", "supervisord.conf"]
