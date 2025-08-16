#!/usr/bin/env python3
"""
Startup script to initialize the application
This ensures models are trained before the API/UI starts
"""

import sys
import os
import logging

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_file():
    """Check if data file exists"""
    from src.config import CSV_PATH
    if not os.path.exists(CSV_PATH):
        logger.error(f"Data file not found: {CSV_PATH}")
        # Create sample data file if none exists
        create_sample_data()
        return True
    logger.info(f"Data file found: {CSV_PATH}")
    return True

def create_sample_data():
    """Create sample data if no data file exists"""
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    
    logger.info("Creating sample data...")
    
    # Generate sample transaction data
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
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_transactions.csv', index=False)
    logger.info(f"Created sample data with {len(df)} transactions for {len(df['customer_id'].unique())} customers")

def check_models():
    """Check if trained models exist"""
    from src.config import XGB_MODEL, SCALER_F, FEATURES_F, KM_MODEL_F
    
    model_files = [XGB_MODEL, SCALER_F, FEATURES_F, KM_MODEL_F]
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        logger.info(f"Missing model files: {missing_models}")
        return False
    
    logger.info("All model files found")
    return True

def train_models():
    """Train the models"""
    try:
        logger.info("Starting model training...")
        from src.train_pipeline import main as train_main
        train_main()
        logger.info("Model training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("=== Application Startup ===")
    
    try:
        # Check data file
        if not check_data_file():
            logger.error("Data check failed")
            return False
        
        # Check if models exist, train if not
        if not check_models():
            logger.info("Models not found, starting training...")
            if not train_models():
                logger.error("Failed to train models")
                return False
        else:
            logger.info("Models already exist, skipping training")
        
        logger.info("=== Startup completed successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
