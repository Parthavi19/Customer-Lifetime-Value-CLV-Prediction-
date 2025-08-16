FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy critical application files explicitly
COPY ui/
COPY src/ 

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts ui src

# Verify critical file exists
RUN ls -la /app/ui/app_streamlit.py || (echo "Critical file missing" && exit 1)

# Create a non-root user for better security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run Streamlit directly
CMD ["streamlit", "run", "/app/ui/app_streamlit.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]

