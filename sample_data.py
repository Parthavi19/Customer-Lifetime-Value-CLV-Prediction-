import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_data():
    """Create sample transaction data for testing"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data
    n_customers = 1000
    n_transactions = 5000
    
    # Date range: last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    data = []
    
    for _ in range(n_transactions):
        customer_id = np.random.randint(1, n_customers + 1)
        
        # Generate random transaction date
        random_days = np.random.randint(0, 730)
        transaction_date = start_date + timedelta(days=random_days)
        
        # Generate transaction details
        transaction_id = f"TXN_{np.random.randint(100000, 999999)}"
        quantity = np.random.randint(1, 5)
        unit_price = np.round(np.random.uniform(10, 200), 2)
        total_amount = quantity * unit_price
        
        data.append({
            'transaction_date': transaction_date.strftime('%Y-%m-%d'),
            'customer_id': customer_id,
            'transaction_id': transaction_id,
            'quantity_purchased': quantity,
            'unit_price': unit_price,
            'total_sale_amount': total_amount
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('transaction_date').reset_index(drop=True)
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Save to CSV
    csv_path = "data/sample_transactions.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Sample data created: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    create_sample_data()
