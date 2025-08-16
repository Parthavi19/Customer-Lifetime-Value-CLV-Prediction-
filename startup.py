#!/usr/bin/env python3
"""
Enhanced startup script for CLV Dashboard
Handles environment setup and compatibility issues
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def install_compatible_packages():
    """Install compatible package versions"""
    logger.info("Installing compatible packages...")
    
    compatible_packages = [
        "numpy==1.23.5",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "shap==0.41.0",
        "streamlit==1.27.0",
        "joblib==1.3.2",
        "matplotlib==3.7.2"
    ]
    
    try:
        for package in compatible_packages:
            logger.info(f"Installing {package}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--upgrade"
            ], capture_output=True, text=True, check=True)
            
        logger.info("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Package installation failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_data_directory():
    """Check and create data directory"""
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory: {data_dir}")
    
    # Check for CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("⚠️  No CSV files found in data/ directory")
        logger.info("Please add your transaction data CSV file to the data/ directory")
        return False
    
    logger.info(f"✅ Found CSV file(s): {[f.name for f in csv_files]}")
    return True

def create_artifacts_directory():
    """Create artifacts directory for models"""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts directory ready: {artifacts_dir}")
    return True

def test_imports():
    """Test critical imports"""
    logger.info("Testing critical imports...")
    
    critical_imports = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn", None),
        ("xgboost", "xgb"),
        ("joblib", None),
        ("matplotlib.pyplot", "plt")
    ]
    
    failed_imports = []
    
    for module, alias in critical_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            logger.info(f"✅ {module} - OK")
        except ImportError as e:
            logger.error(f"❌ {module} - FAILED: {e}")
            failed_imports.append(module)
    
    # Test SHAP separately (optional)
    try:
        import shap
        logger.info(f"✅ SHAP {shap.__version__} - OK")
    except ImportError as e:
        logger.warning(f"⚠️  SHAP - OPTIONAL: {e}")
    
    return len(failed_imports) == 0, failed_imports

def run_training_pipeline():
    """Run the training pipeline if needed"""
    logger.info("Checking if training pipeline needs to be run...")
    
    required_files = [
        "artifacts/xgb_model.joblib",
        "artifacts/customer_features.csv",
        "artifacts/scaler.joblib",
        "artifacts/feature_cols.joblib"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.info(f"Missing files: {missing_files}")
        logger.info("Running training pipeline...")
        
        try:
            # Try train_pipeline.py first
            if os.path.exists("train_pipeline.py"):
                result = subprocess.run([
                    sys.executable, "train_pipeline.py"
                ], capture_output=True, text=True, check=True)
                logger.info("✅ Training pipeline completed successfully")
                return True
            # Fallback to src.train_pipeline
            elif os.path.exists("src/train_pipeline.py"):
                result = subprocess.run([
                    sys.executable, "-m", "src.train_pipeline"
                ], capture_output=True, text=True, check=True)
                logger.info("✅ Training pipeline completed successfully")
                return True
            else:
                logger.error("❌ No training pipeline found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            if e.stdout:
                logger.info(f"Standard output: {e.stdout}")
            return False
    else:
        logger.info("✅ All required files exist - training pipeline not needed")
        return True

def create_sample_data():
    """Create sample data if no data exists"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    data_dir = Path("data")
    sample_file = data_dir / "sample_transactions.csv"
    
    if sample_file.exists():
        logger.info("Sample data already exists")
        return True
    
    logger.info("Creating sample transaction data...")
    
    # Generate sample data
    np.random.seed(42)
    n_customers = 1000
    n_transactions = 5000
    
    start_date = datetime.now() - timedelta(days=365)
    
    # Generate sample transactions
    data = []
    for i in range(n_transactions):
        customer_id = np.random.randint(1000, 1000 + n_customers)
        transaction_date = start_date + timedelta(days=np.random.randint(0, 365))
        transaction_id = f"TXN_{i+1:06d}"
        quantity = np.random.randint(1, 5)
        unit_price = np.random.uniform(10, 200)
        total_amount = quantity * unit_price
        
        data.append({
            'customer_id': customer_id,
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'transaction_id': transaction_id,
            'quantity_purchased': quantity,
            'unit_price': round(unit_price, 2),
            'total_sale_amount': round(total_amount, 2)
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(sample_file, index=False)
    
    logger.info(f"✅ Sample data created: {sample_file}")
    logger.info(f"   - {len(df)} transactions")
    logger.info(f"   - {df['customer_id'].nunique()} customers")
    
    return True

def main():
    """Main startup function"""
    logger.info("🚀 Starting CLV Dashboard initialization...")
    
    # Step 1: Check Python version
    if not check_python_version():
        logger.error("❌ Python version check failed")
        return False
    
    # Step 2: Create directories
    create_artifacts_directory()
    
    # Step 3: Check for data
    if not check_data_directory():
        logger.info("No data found, creating sample data...")
        if not create_sample_data():
            logger.error("❌ Failed to create sample data")
            return False
    
    # Step 4: Install compatible packages
    logger.info("Checking package compatibility...")
    try:
        if not install_compatible_packages():
            logger.error("❌ Package installation failed")
            return False
    except Exception as e:
        logger.warning(f"Package installation issue: {e}")
    
    # Step 5: Test imports
    imports_ok, failed = test_imports()
    if not imports_ok:
        logger.error(f"❌ Critical imports failed: {failed}")
        return False
    
    # Step 6: Run training pipeline
    if not run_training_pipeline():
        logger.error("❌ Training pipeline failed")
        return False
    
    logger.info("✅ CLV Dashboard initialization completed successfully!")
    logger.info("🎯 You can now run: streamlit run app_streamlit.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
