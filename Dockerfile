FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data artifacts src ui

# Copy application files in correct structure
COPY app_streamlit.py ./ui/app_streamlit.py
COPY api.py ./src/api.py
COPY src/ ./src/
COPY __init__.py ./src/__init__.py

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting application..."\n\
echo "Files in /app:"\n\
ls -la /app/\n\
echo "Files in /app/src:"\n\
ls -la /app/src/\n\
echo "Files in /app/ui:"\n\
ls -la /app/ui/\n\
\n\
# Check if training is needed\n\
if [ ! -f "/app/artifacts/xgb_model.joblib" ]; then\n\
    echo "Training model..."\n\
    cd /app && python -m src.train_pipeline || echo "Training failed, continuing..."\n\
fi\n\
\n\
# Start Streamlit\n\
echo "Starting Streamlit..."\n\
cd /app && streamlit run ui/app_streamlit.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false\n\
' > /app/start.sh && chmod +x /app/start.sh

# Create a non-root user for better security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Use the startup script
CMD ["/app/start.sh"]
