FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Create startup script that handles your actual file structure
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Starting CLV Dashboard ==="\n\
echo "Current directory: $(pwd)"\n\
echo "Files in /app:"\n\
ls -la /app/\n\
echo "Files in /app/ui:"\n\
ls -la /app/ui/ || echo "ui directory not found"\n\
echo "Files in /app/src:"\n\
ls -la /app/src/ || echo "src directory not found"\n\
\n\
# Check if we need to initialize\n\
if [ ! -f "artifacts/xgb_model.joblib" ]; then\n\
    echo "Models not found, running initialization..."\n\
    if [ -f "startup.py" ]; then\n\
        python startup.py || echo "Initialization completed with warnings"\n\
    else\n\
        echo "Creating sample data and training models..."\n\
        python -c "\n\
import sys, os\n\
sys.path.insert(0, os.path.abspath(\".\"))\n\
try:\n\
    from src.train_pipeline import main\n\
    main()\n\
    print(\"Training completed successfully\")\n\
except Exception as e:\n\
    print(f\"Training failed: {e}\")\n\
    # Create sample data if training fails\n\
    import pandas as pd\n\
    import numpy as np\n\
    from datetime import datetime, timedelta\n\
    \n\
    np.random.seed(42)\n\
    n_customers = 1000\n\
    n_transactions = 5000\n\
    \n\
    customer_ids = [f\"CUST_{i:04d}\" for i in range(1, n_customers + 1)]\n\
    data = []\n\
    start_date = datetime.now() - timedelta(days=365*2)\n\
    \n\
    for i in range(n_transactions):\n\
        customer_id = np.random.choice(customer_ids)\n\
        transaction_date = start_date + timedelta(days=np.random.randint(0, 365*2))\n\
        quantity = np.random.randint(1, 5)\n\
        unit_price = np.random.uniform(10, 200)\n\
        total_amount = quantity * unit_price\n\
        \n\
        data.append({\n\
            \"transaction_id\": f\"TXN_{i:06d}\",\n\
            \"customer_id\": customer_id,\n\
            \"transaction_date\": transaction_date.strftime(\"%Y-%m-%d %H:%M:%S\"),\n\
            \"quantity_purchased\": quantity,\n\
            \"unit_price\": round(unit_price, 2),\n\
            \"total_sale_amount\": round(total_amount, 2)\n\
        })\n\
    \n\
    df = pd.DataFrame(data)\n\
    os.makedirs(\"data\", exist_ok=True)\n\
    df.to_csv(\"data/sample_transactions.csv\", index=False)\n\
    print(f\"Created sample data with {len(df)} transactions\")\n\
    \n\
    # Try training again\n\
    try:\n\
        from src.train_pipeline import main\n\
        main()\n\
        print(\"Training completed after creating sample data\")\n\
    except Exception as e2:\n\
        print(f\"Training still failed: {e2}\")\n\
"\n\
    fi\n\
else\n\
    echo "Models found, skipping initialization"\n\
fi\n\
\n\
echo "Starting Streamlit dashboard..."\n\
echo "Running: streamlit run ui/app_streamlit.py"\n\
streamlit run ui/app_streamlit.py \\\n\
    --server.port=8080 \\\n\
    --server.address=0.0.0.0 \\\n\
    --server.headless=true \\\n\
    --server.fileWatcherType=none \\\n\
    --browser.gatherUsageStats=false\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Start the application
CMD ["/app/start.sh"]
