"""
Synthetic Financial Transaction Dataset Generator
=================================================
Generates a hyper-realistic synthetic dataset for fraud detection
with realistic class imbalance (~2% fraud rate), temporal patterns,
behavioral features, and multi-dimensional fraud signatures.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
SEED = 42
N_TRANSACTIONS = 200_000
FRAUD_RATE = 0.02  # 2% — realistic for financial fraud

MERCHANT_CATEGORIES = [
    'grocery', 'electronics', 'restaurant', 'gas_station', 'online_retail',
    'travel', 'entertainment', 'healthcare', 'utilities', 'atm_withdrawal',
    'jewelry', 'cash_advance', 'money_transfer', 'gambling', 'subscription'
]

CARD_TYPES = ['visa', 'mastercard', 'amex', 'discover']
ENTRY_MODES = ['chip', 'swipe', 'contactless', 'online', 'manual']
COUNTRIES = ['US', 'US', 'US', 'US', 'US', 'UK', 'CA', 'DE', 'FR', 'IN',
             'BR', 'JP', 'AU', 'NG', 'RU', 'CN', 'MX']  # US heavily weighted

CITIES_BY_COUNTRY = {
    'US': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
           'San Francisco', 'Miami', 'Seattle', 'Denver', 'Atlanta'],
    'UK': ['London', 'Manchester', 'Birmingham'],
    'CA': ['Toronto', 'Vancouver', 'Montreal'],
    'DE': ['Berlin', 'Munich', 'Frankfurt'],
    'FR': ['Paris', 'Lyon', 'Marseille'],
    'IN': ['Mumbai', 'Delhi', 'Bangalore'],
    'BR': ['Sao Paulo', 'Rio de Janeiro'],
    'JP': ['Tokyo', 'Osaka'],
    'AU': ['Sydney', 'Melbourne'],
    'NG': ['Lagos', 'Abuja'],
    'RU': ['Moscow', 'St Petersburg'],
    'CN': ['Shanghai', 'Beijing'],
    'MX': ['Mexico City', 'Guadalajara']
}


def _generate_timestamps(n: int, rng: np.random.Generator) -> pd.Series:
    """Generate realistic transaction timestamps over 6 months."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 6, 30)
    total_seconds = int((end - start).total_seconds())
    random_seconds = rng.integers(0, total_seconds, size=n)
    timestamps = pd.to_datetime([start + timedelta(seconds=int(s)) for s in sorted(random_seconds)])
    return timestamps


def _assign_customer_profiles(n: int, n_customers: int, rng: np.random.Generator) -> dict:
    """Create customer profiles with spending habits."""
    customer_ids = rng.integers(10000, 10000 + n_customers, size=n)

    # Each customer has a "home country" and avg spending pattern
    unique_customers = np.unique(customer_ids)
    customer_home = {c: rng.choice(COUNTRIES[:5]) for c in unique_customers}  # mostly US
    customer_avg_amount = {c: rng.lognormal(mean=3.5, sigma=0.8) for c in unique_customers}
    customer_card = {c: rng.choice(CARD_TYPES) for c in unique_customers}

    return {
        'customer_ids': customer_ids,
        'customer_home': customer_home,
        'customer_avg_amount': customer_avg_amount,
        'customer_card': customer_card,
    }


def generate_dataset(n_transactions: int = N_TRANSACTIONS,
                     fraud_rate: float = FRAUD_RATE,
                     save_path: str = None,
                     seed: int = SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic financial transaction dataset.

    Parameters
    ----------
    n_transactions : int
        Total number of transactions to generate.
    fraud_rate : float
        Fraction of fraudulent transactions (0.0 to 1.0).
    save_path : str, optional
        Path to save the CSV file.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated dataset with fraud labels.
    """
    rng = np.random.default_rng(seed)
    n_customers = int(n_transactions * 0.05)  # ~5% unique customers

    print(f"[DATA] Generating {n_transactions:,} transactions with {fraud_rate*100:.1f}% fraud rate...")

    # --- Core transaction fields ---
    timestamps = _generate_timestamps(n_transactions, rng)
    profiles = _assign_customer_profiles(n_transactions, n_customers, rng)

    customer_ids = profiles['customer_ids']
    countries = np.array([
        profiles['customer_home'].get(c, rng.choice(COUNTRIES))
        for c in customer_ids
    ])
    cities = np.array([
        rng.choice(CITIES_BY_COUNTRY.get(co, ['Unknown']))
        for co in countries
    ])
    card_types = np.array([profiles['customer_card'].get(c, 'visa') for c in customer_ids])

    # Transaction amounts — lognormal distribution per customer
    amounts = np.array([
        max(0.50, rng.lognormal(mean=np.log(profiles['customer_avg_amount'].get(c, 50)), sigma=0.6))
        for c in customer_ids
    ])
    amounts = np.round(amounts, 2)

    merchant_categories = rng.choice(MERCHANT_CATEGORIES, size=n_transactions)
    entry_modes = rng.choice(ENTRY_MODES, size=n_transactions, p=[0.30, 0.20, 0.20, 0.25, 0.05])

    # --- Time-based features ---
    hours = np.array([t.hour for t in timestamps])
    days_of_week = np.array([t.dayofweek for t in timestamps])
    is_weekend = (days_of_week >= 5).astype(int)
    is_night = ((hours >= 22) | (hours <= 5)).astype(int)

    # --- Build initial DataFrame ---
    df = pd.DataFrame({
        'transaction_id': [f'TXN-{i:08d}' for i in range(n_transactions)],
        'timestamp': timestamps,
        'customer_id': [f'CUST-{c}' for c in customer_ids],
        'amount': amounts,
        'merchant_category': merchant_categories,
        'card_type': card_types,
        'entry_mode': entry_modes,
        'country': countries,
        'city': cities,
        'hour_of_day': hours,
        'day_of_week': days_of_week,
        'is_weekend': is_weekend,
        'is_night': is_night,
    })

    # --- Behavioral / Aggregated features ---
    # Sort by customer and timestamp for rolling calculations
    df = df.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)

    # Transaction frequency: count of transactions in the same customer group
    df['customer_txn_count'] = df.groupby('customer_id').cumcount() + 1

    # Rolling average amount (per customer, last 5 txns)
    df['rolling_avg_amount'] = (
        df.groupby('customer_id')['amount']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Amount deviation from customer's rolling average
    df['amount_deviation'] = ((df['amount'] - df['rolling_avg_amount']) /
                               df['rolling_avg_amount'].clip(lower=1)).round(4)

    # Time since last transaction (in hours)
    df['time_since_last_txn'] = (
        df.groupby('customer_id')['timestamp']
        .diff()
        .dt.total_seconds()
        .fillna(9999)
        / 3600  # convert to hours
    ).round(2)

    # Velocity: approximate transactions in last 24h window
    # Use cumcount within customer as a base, capped at 15, with random noise
    cumcounts = df.groupby('customer_id').cumcount().values
    df['velocity_24h'] = np.clip(cumcounts, 0, 15) + rng.integers(0, 3, size=n_transactions)

    # Cross-border flag: is this transaction in a different country than customer's home?
    customer_home_map = {f'CUST-{k}': v for k, v in profiles['customer_home'].items()}
    df['home_country'] = df['customer_id'].map(customer_home_map)
    df['is_cross_border'] = (df['country'] != df['home_country']).astype(int)

    # Distance from home (synthetic — normalized 0-1)
    df['distance_from_home'] = np.where(
        df['is_cross_border'] == 1,
        rng.uniform(0.5, 1.0, size=n_transactions),
        rng.uniform(0.0, 0.3, size=n_transactions)
    ).round(4)

    # --- Assign fraud labels with realistic patterns ---
    n_fraud = int(n_transactions * fraud_rate)
    fraud_labels = np.zeros(n_transactions, dtype=int)

    # Calculate fraud propensity scores for each transaction
    propensity = np.zeros(n_transactions)

    # High amounts increase fraud likelihood
    amount_percentile = np.percentile(df['amount'], 95)
    propensity += np.where(df['amount'] > amount_percentile, 3.0, 0.0)
    propensity += np.where(df['amount'] > np.percentile(df['amount'], 99), 5.0, 0.0)

    # Night transactions more likely fraud
    propensity += np.where(df['is_night'] == 1, 2.0, 0.0)

    # Cross-border increases fraud risk
    propensity += np.where(df['is_cross_border'] == 1, 2.5, 0.0)

    # High velocity (many recent txns)
    propensity += np.where(df['velocity_24h'] > 8, 2.0, 0.0)

    # Manual entry mode — higher risk
    propensity += np.where(df['entry_mode'] == 'manual', 3.0, 0.0)
    propensity += np.where(df['entry_mode'] == 'online', 1.0, 0.0)

    # Risky merchant categories
    risky_categories = ['cash_advance', 'money_transfer', 'gambling', 'jewelry']
    propensity += np.where(df['merchant_category'].isin(risky_categories), 2.5, 0.0)

    # Large deviation from average spending
    propensity += np.where(df['amount_deviation'] > 2.0, 2.0, 0.0)

    # Short time since last transaction
    propensity += np.where(df['time_since_last_txn'] < 0.5, 1.5, 0.0)

    # Add noise
    propensity += rng.normal(0, 1, size=n_transactions)

    # Select top-N propensity scores as fraud
    fraud_indices = np.argsort(propensity)[-n_fraud:]
    fraud_labels[fraud_indices] = 1

    df['is_fraud'] = fraud_labels

    # --- Post-fraud adjustments: make fraud transactions look more suspicious ---
    fraud_mask = df['is_fraud'] == 1

    # Amplify amounts for some fraud transactions
    amplify_mask = fraud_mask & (rng.random(n_transactions) > 0.4)
    df.loc[amplify_mask, 'amount'] = df.loc[amplify_mask, 'amount'] * rng.uniform(2, 8, size=amplify_mask.sum())
    df['amount'] = df['amount'].round(2)

    # Some fraud happens at unusual hours
    night_shift_mask = fraud_mask & (rng.random(n_transactions) > 0.6)
    df.loc[night_shift_mask, 'hour_of_day'] = rng.choice([0, 1, 2, 3, 4, 23], size=night_shift_mask.sum())
    df.loc[night_shift_mask, 'is_night'] = 1

    # --- Final cleanup ---
    # Drop helper columns
    df = df.drop(columns=['home_country'])

    # Reorder columns
    column_order = [
        'transaction_id', 'timestamp', 'customer_id', 'amount',
        'merchant_category', 'card_type', 'entry_mode', 'country', 'city',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
        'customer_txn_count', 'rolling_avg_amount', 'amount_deviation',
        'time_since_last_txn', 'velocity_24h', 'is_cross_border',
        'distance_from_home', 'is_fraud'
    ]
    df = df[column_order].reset_index(drop=True)

    # --- Save ---
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'transactions.csv')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DATA] Dataset saved to {save_path}")
    print(f"[DATA] Shape: {df.shape}")
    print(f"[DATA] Fraud distribution:\n{df['is_fraud'].value_counts(normalize=True).to_string()}")

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary of the dataset for dashboarding."""
    fraud = df[df['is_fraud'] == 1]
    legit = df[df['is_fraud'] == 0]

    return {
        'total_transactions': len(df),
        'total_fraud': len(fraud),
        'total_legitimate': len(legit),
        'fraud_rate': len(fraud) / len(df) * 100,
        'total_amount': df['amount'].sum(),
        'fraud_amount': fraud['amount'].sum(),
        'legitimate_amount': legit['amount'].sum(),
        'avg_fraud_amount': fraud['amount'].mean(),
        'avg_legit_amount': legit['amount'].mean(),
        'median_fraud_amount': fraud['amount'].median(),
        'median_legit_amount': legit['amount'].median(),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'n_customers': df['customer_id'].nunique(),
        'top_fraud_categories': fraud['merchant_category'].value_counts().head(5).to_dict(),
        'top_fraud_countries': fraud['country'].value_counts().head(5).to_dict(),
        'fraud_by_hour': fraud['hour_of_day'].value_counts().sort_index().to_dict(),
        'fraud_by_entry_mode': fraud['entry_mode'].value_counts().to_dict(),
    }


if __name__ == '__main__':
    df = generate_dataset()
    summary = get_dataset_summary(df)
    print(f"\n{'='*50}")
    print(f"Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
