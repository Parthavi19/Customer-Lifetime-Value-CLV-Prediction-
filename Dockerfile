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

# Copy critical application files explicitly
COPY api.py .
COPY ui/ ui/
COPY src/ src/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts ui src

# Verify critical files exist
RUN ls -l /app/api.py /app/ui/app_streamlit.py || (echo "Critical files missing" && exit 1)

# Create a non-root user for better security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Use supervisord to manage both FastAPI and Streamlit
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
