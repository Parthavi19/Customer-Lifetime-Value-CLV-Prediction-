import pandas as pd
from datetime import datetime
from .config import CSV_PATH, DATE_COL, CUST_COL, TOTAL_COL
import logging

logger = logging.getLogger(__name__)

def load_and_clean():
    """Load and clean the transaction data"""
    try:
        logger.info(f"Loading data from {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        
        # Convert date column to datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        
        # Remove rows with null values in key columns
        key_cols = [DATE_COL, CUST_COL, TOTAL_COL]
        df = df.dropna(subset=key_cols)
        
        # Remove rows with zero or negative total amounts
        df = df[df[TOTAL_COL] > 0]
        
        logger.info(f"Loaded {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
