"""
FraudShield AI - Data Generation Module.
Generates hyper-realistic synthetic financial transaction data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants required by app.py
MERCHANT_CATEGORIES = [
    "Retail", "Entertainment", "Food & Dining", "Travel", 
    "Health", "Services", "Online Shopping", "Electronics", 
    "Utilities", "Others"
]
CARD_TYPES = ["Visa", "Mastercard", "Amex", "Discover"]
ENTRY_MODES = ["Chip", "Magnetic Stripe", "Contactless", "Online"]
COUNTRIES = ["USA", "UK", "Canada", "Germany", "France", "India", "Australia", "Japan"]

def generate_dataset(n_samples=200000, save_path=None):
    """Generate synthetic transactions with realistic fraud patterns."""
    np.random.seed(42)
    
    # Base features
    timestamps = [datetime(2024, 1, 1) + timedelta(minutes=int(i*1.5)) for i in range(n_samples)]
    
    data = pd.DataFrame({
        'transaction_id': [f'TXN-{i:08d}' for i in range(n_samples)],
        'timestamp': timestamps,
        'customer_id': np.random.randint(1000, 5000, n_samples),
        'amount': np.random.exponential(scale=100, size=n_samples) + np.random.normal(loc=20, scale=10, size=n_samples),
        'merchant_category': np.random.choice(MERCHANT_CATEGORIES, n_samples),
        'card_type': np.random.choice(CARD_TYPES, n_samples),
        'entry_mode': np.random.choice(ENTRY_MODES, n_samples),
        'country': np.random.choice(COUNTRIES, n_samples),
        'is_fraud': 0
    })
    
    # Clip amount
    data['amount'] = data['amount'].clip(lower=0.5).round(2)
    
    # ── Inject Fraud Patterns ──────────────────────────────────────────
    # 1. High amount transactions are more likely to be fraud
    high_amount_mask = data['amount'] > 1000
    data.loc[data[high_amount_mask].sample(frac=0.15).index, 'is_fraud'] = 1
    
    # 2. Specific merchant categories (e.g., Electronics, Travel)
    risky_cats = ["Electronics", "Travel", "Online Shopping"]
    cat_mask = data['merchant_category'].isin(risky_cats)
    data.loc[data[cat_mask].sample(frac=0.03).index, 'is_fraud'] = 1
    
    # 3. Night-time transactions (temporal pattern)
    data['hour'] = data['timestamp'].dt.hour
    night_mask = (data['hour'] >= 0) & (data['hour'] <= 4)
    data.loc[data[night_mask].sample(frac=0.05).index, 'is_fraud'] = 1
    
    # 4. Entry mode - Online is riskier
    online_mask = data['entry_mode'] == "Online"
    data.loc[data[online_mask].sample(frac=0.04).index, 'is_fraud'] = 1
    
    # Final cleanup
    data = data.drop(columns=['hour'])
    
    if save_path:
        data.to_csv(save_path, index=False)
        
    return data

def get_dataset_summary(df):
    """Calculate summary statistics for the executive dashboard."""
    fraud_df = df[df['is_fraud'] == 1]
    legit_df = df[df['is_fraud'] == 0]
    
    return {
        'total_transactions': len(df),
        'total_fraud': len(fraud_df),
        'total_legitimate': len(legit_df),
        'fraud_rate': (len(fraud_df) / len(df)) * 100,
        'total_amount': df['amount'].sum(),
        'fraud_amount': fraud_df['amount'].sum(),
        'avg_fraud_amount': fraud_df['amount'].mean(),
        'avg_legit_amount': legit_df['amount'].mean(),
        'n_customers': df['customer_id'].nunique(),
        'date_range': (df['timestamp'].min(), df['timestamp'].max())
    }
