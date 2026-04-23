"""
FraudShield AI - Preprocessing Module.
Handles feature engineering, encoding, scaling, and SMOTE resampling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.over_sampling import SMOTE

def preprocess_pipeline(df):
    """Full data preprocessing pipeline for training/testing."""
    df = df.copy()
    
    # 1. Temporal Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Behavioral Features
    # Transaction frequency per customer (rolling count)
    df = df.sort_values(['customer_id', 'timestamp'])
    df['customer_tx_count'] = df.groupby('customer_id').cumcount()
    
    # 3. Encoding Categorical Features
    cat_cols = ['merchant_category', 'card_type', 'entry_mode', 'country']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # 4. Prepare X and y
    drop_cols = ['transaction_id', 'timestamp', 'customer_id', 'is_fraud']
    X = df.drop(columns=drop_cols)
    y = df['is_fraud']
    
    feature_names = X.columns.tolist()
    
    # 5. Scaling (Robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # 6. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 7. SMOTE Resampling (Training set only!)
    pre_smote_dist = y_train.value_counts().to_dict()
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    post_smote_dist = pd.Series(y_train_resampled).value_counts().to_dict()
    
    return {
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'encoders': encoders,
        'scaler': scaler,
        'pre_smote_distribution': pre_smote_dist,
        'post_smote_distribution': post_smote_dist
    }

def preprocess_single_transaction(txn_dict, pipeline_assets):
    """Preprocess a single incoming transaction for prediction."""
    # Convert to DF
    df = pd.DataFrame([txn_dict])
    
    # 1. Temporal
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        df['hour'] = ts.dt.hour
        df['day_of_week'] = ts.dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Placeholder for behavioral (would normally fetch from DB)
    df['customer_tx_count'] = 10 
    
    # 2. Encoding
    encoders = pipeline_assets['encoders']
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unknown labels by assigning a default or the first label
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0
                
    # 3. Scaling
    feature_names = pipeline_assets['feature_names']
    X = df[feature_names]
    X_scaled = pipeline_assets['scaler'].transform(X)
    
    return X_scaled
