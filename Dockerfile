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
COPY ui/ ui/
COPY src/ src/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts ui src

# Verify critical files and supervisord.conf exist
RUN ls -la /app/ui/app_streamlit.py /app/src/api.py /etc/supervisor/conf.d/supervisord.conf || (echo "Critical files or config missing" && exit 1)
RUN cat /etc/supervisor/conf.d/supervisord.conf

# Create a non-root user for better security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Use supervisord to manage both FastAPI and Streamlit
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
