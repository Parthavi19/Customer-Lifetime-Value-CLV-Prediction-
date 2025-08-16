#!/usr/bin/env python3
"""
Startup script for CLV Dashboard
Handles initialization for the ui/ and src/ file structure
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample transaction data"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    logger.info("Creating sample transaction data...")
    
    np.random.seed(42)
    n_customers = 1000
    n_transactions = 5000
    
    # Create customers
    customer_ids = [f"CUST_{i:04d}" for i in range(1, n_customers + 1)]
    
    # Generate transactions
    data = []
    start_date = datetime.now() - timedelta(days=365*2)
    
    for i in range(n_transactions):
        customer_id = np.random.choice(customer_ids)
        transaction_date = start_date + timedelta(
            days=np.random.randint(0, 365*2),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        quantity = np.random.randint(1, 5)
        unit_price = np.random.uniform(10, 200)
        total_amount = quantity * unit_price
        
        data.append({
            'transaction_id': f"TXN_{i:06d}",
            'customer_id': customer_id,
            'transaction_date': transaction_date.strftime('%Y-%m-%d %H:%M:%S'),
            'quantity_purchased': quantity,
            'unit_price': round(unit_price, 2),
            'total_sale_amount': round(total_amount, 2)
        })
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    csv_path = data_dir / 'sample_transactions.csv'
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Created sample data: {csv_path}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Unique customers: {df['customer_id'].nunique()}")
    
    return csv_path

def check_data():
    """Check if data file exists"""
    try:
        # Try to import config and check configured path
        from src.config import CSV_PATH
        if os.path.exists(CSV_PATH):
            logger.info(f"Data file found: {CSV_PATH}")
            return True
    except ImportError:
        logger.warning("Could not import config, checking default locations")
    
    # Check common locations
    data_dir = project_root / 'data'
    csv_files = list(data_dir.glob('*.csv')) if data_dir.exists() else []
    
    if csv_files:
        logger.info(f"Found CSV files: {csv_files}")
        return True
    
    logger.info("No data files found, will create sample data")
    return False

def check_models():
    """Check if trained models exist"""
    artifacts_dir = project_root / 'artifacts'
    
    required_files = [
        'xgb_model.joblib',
        'scaler.joblib',
        'feature_cols.joblib',
        'kmeans_model.joblib'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = artifacts_dir / file
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        logger.info(f"Missing model files: {missing_files}")
        return False
    
    logger.info("All model files found")
    return True

def train_models():
    """Train the models using the pipeline"""
    try:
        logger.info("Starting model training pipeline...")
        from src.train_pipeline import main as train_main
        train_main()
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.exception("Training error details:")
        return False

def main():
    """Main startup function"""
    logger.info("=== CLV Dashboard Startup ===")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path[:3]}")
    
    try:
        # Ensure artifacts directory exists
        artifacts_dir = project_root / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        
        # Check data
        if not check_data():
            logger.info("Creating sample data...")
            create_sample_data()
        
        # Check models
        if not check_models():
            logger.info("Training models...")
            if not train_models():
                logger.error("Failed to train models")
                return False
        else:
            logger.info("Models already exist, skipping training")
        
        logger.info("=== Startup completed successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.exception("Startup error details:")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
