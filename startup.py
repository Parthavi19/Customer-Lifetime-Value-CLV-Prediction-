#!/usr/bin/env python3
"""
Startup script for Cloud Run deployment
Ensures data exists and models are trained before starting the API
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data_exists():
    """Ensure sample data exists if no data file is found"""
    try:
        from src.config import CSV_PATH
        if not CSV_PATH.exists():
            logger.info("No data file found, creating sample data...")
            from create_sample_data import create_sample_data
            create_sample_data()
    except Exception as e:
        logger.error(f"Error ensuring data exists: {e}")
        # Create minimal sample data as fallback
        create_minimal_sample_data()

def create_minimal_sample_data():
    """Create minimal sample data as fallback"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    Path("data").mkdir(exist_ok=True)
    
    # Minimal dataset
    data = []
    for i in range(100):
        data.append({
            'transaction_date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
            'customer_id': i % 20 + 1,
            'transaction_id': f'TXN_{1000 + i}',
            'quantity_purchased': 1,
            'unit_price': 50.0,
            'total_sale_amount': 50.0
        })
    
    df = pd.DataFrame(data)
    df.to_csv("data/sample_transactions.csv", index=False)
    logger.info("Created minimal sample data")

def check_and_train_models():
    """Check if models exist, train if they don't"""
    try:
        from src.config import XGB_MODEL, KM_MODEL_F, CUSTOMER_FEATS
        
        # Check if models exist
        if not (XGB_MODEL.exists() and KM_MODEL_F.exists() and CUSTOMER_FEATS.exists()):
            logger.info("Models not found, training...")
            from src.train_pipeline import main as train_main
            train_main()
            logger.info("Training completed successfully")
        else:
            logger.info("Models already exist, skipping training")
            
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        # Create dummy models for basic API functionality
        create_dummy_artifacts()

def create_dummy_artifacts():
    """Create dummy artifacts to prevent app crashes"""
    from pathlib import Path
    import joblib
    import pandas as pd
    
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Create dummy feature list
    features = ['recency_days', 'frequency', 'monetary', 'avg_order_value', 'std_order_value', 'avg_days_between_txn']
    joblib.dump(features, artifacts_dir / "feature_cols.joblib")
    
    # Create dummy customer features CSV
    dummy_df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'predicted_clv': [100, 200, 150],
        'segment_label': [0, 1, 0],
        'segment_name': ['Low Value', 'High Value', 'Low Value']
    })
    dummy_df.to_csv(artifacts_dir / "customer_features.csv", index=False)
    
    logger.info("Created dummy artifacts for basic functionality")

def main():
    """Main startup sequence"""
    logger.info("Starting application initialization...")
    
    # Ensure data exists
    ensure_data_exists()
    
    # Check and train models
    check_and_train_models()
    
    logger.info("Application initialization completed")

if __name__ == "__main__":
    main()
