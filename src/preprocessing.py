"""
Data Preprocessing & Feature Engineering Pipeline
==================================================
Handles feature engineering, encoding, scaling, train/test splitting,
and SMOTE resampling for the fraud detection pipeline.

Critical design decision: SMOTE is applied ONLY to the training set.
The test set is never touched — it reflects real-world class distribution.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# Features to exclude from model training
EXCLUDE_COLS = ['transaction_id', 'timestamp', 'customer_id', 'city', 'is_fraud']

# Categorical columns to encode
CATEGORICAL_COLS = ['merchant_category', 'card_type', 'entry_mode', 'country']

# Numerical columns to scale
NUMERICAL_COLS = [
    'amount', 'hour_of_day', 'day_of_week', 'customer_txn_count',
    'rolling_avg_amount', 'amount_deviation', 'time_since_last_txn',
    'velocity_24h', 'distance_from_home'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features for improved model performance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with additional engineered features.
    """
    df = df.copy()

    # --- Interaction features ---
    # Amount * distance interaction (fraud often = high amount + far from home)
    df['amount_x_distance'] = (df['amount'] * df['distance_from_home']).round(4)

    # Amount * night interaction
    df['amount_x_night'] = (df['amount'] * df['is_night']).round(2)

    # Velocity * amount deviation (rapid + unusual = suspicious)
    df['velocity_x_deviation'] = (df['velocity_24h'] * df['amount_deviation'].abs()).round(4)

    # Cross-border * night (international night transactions)
    df['cross_border_x_night'] = df['is_cross_border'] * df['is_night']

    # Log-transformed amount (reduces skew)
    df['log_amount'] = np.log1p(df['amount']).round(4)

    # Amount bins (quartile-based risk buckets)
    df['amount_bucket'] = pd.qcut(df['amount'], q=10, labels=False, duplicates='drop')

    # Hour buckets (business hours vs off-hours)
    df['hour_bucket'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 5, 9, 17, 21, 24],
        labels=['late_night', 'early_morning', 'business_hours', 'evening', 'night']
    ).astype(str)

    # Is high value transaction (> 95th percentile)
    df['is_high_value'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

    # Is rapid succession (< 30 min since last txn)
    df['is_rapid_succession'] = (df['time_since_last_txn'] < 0.5).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with categorical columns.
    encoders : dict, optional
        Pre-fitted encoders (for test set transformation).
    fit : bool
        Whether to fit new encoders or use existing ones.

    Returns
    -------
    tuple
        (encoded DataFrame, encoders dict)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    # Handle hour_bucket as categorical too
    cat_cols = CATEGORICAL_COLS + ['hour_bucket']

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                # Handle unseen categories gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])

    return df, encoders


def preprocess_pipeline(df: pd.DataFrame,
                        test_size: float = 0.2,
                        smote_strategy: float = 0.5,
                        random_state: int = 42) -> dict:
    """
    Full preprocessing pipeline: feature engineering → encoding → scaling →
    train/test split → SMOTE resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataset with 'is_fraud' column.
    test_size : float
        Fraction of data to use as test set.
    smote_strategy : float
        SMOTE sampling strategy (ratio of minority to majority).
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing all pipeline outputs:
        - X_train, y_train: Original training data (before SMOTE)
        - X_train_resampled, y_train_resampled: SMOTE-resampled training data
        - X_test, y_test: Test data (untouched)
        - feature_names: List of feature column names
        - scaler: Fitted RobustScaler
        - encoders: Fitted label encoders
        - class_distribution: Before/after SMOTE distribution
    """
    print("[PREPROCESSING] Starting pipeline...")

    # Step 1: Feature Engineering
    print("[PREPROCESSING] Step 1/5: Feature engineering...")
    df_eng = engineer_features(df)

    # Step 2: Encode categoricals
    print("[PREPROCESSING] Step 2/5: Encoding categoricals...")
    df_enc, encoders = encode_categoricals(df_eng, fit=True)

    # Step 3: Prepare features and target
    feature_cols = [c for c in df_enc.columns if c not in EXCLUDE_COLS]
    X = df_enc[feature_cols].copy()
    y = df_enc['is_fraud'].copy()

    # Fill any NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_names = list(X.columns)
    print(f"[PREPROCESSING] Features ({len(feature_names)}): {feature_names}")

    # Step 4: Train/Test Split (STRATIFIED)
    print("[PREPROCESSING] Step 3/5: Stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[PREPROCESSING] Train set: {X_train.shape[0]:,} samples")
    print(f"[PREPROCESSING] Test set:  {X_test.shape[0]:,} samples")

    # Step 5: Scale numerical features
    print("[PREPROCESSING] Step 4/5: Scaling features (RobustScaler)...")
    scaler = RobustScaler()
    num_cols_present = [c for c in NUMERICAL_COLS if c in feature_names]
    # Also scale engineered numerical columns
    engineered_num = ['amount_x_distance', 'amount_x_night', 'velocity_x_deviation', 'log_amount']
    num_cols_present += [c for c in engineered_num if c in feature_names]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols_present] = scaler.fit_transform(X_train[num_cols_present])
    X_test_scaled[num_cols_present] = scaler.transform(X_test[num_cols_present])

    # Record pre-SMOTE distribution
    pre_smote_dist = y_train.value_counts().to_dict()

    # Step 6: Apply SMOTE (ONLY on training set!)
    print("[PREPROCESSING] Step 5/5: Applying SMOTE to training set...")
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=random_state, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    post_smote_dist = pd.Series(y_train_resampled).value_counts().to_dict()

    print(f"[PREPROCESSING] SMOTE resampling complete:")
    print(f"  Before: {pre_smote_dist}")
    print(f"  After:  {post_smote_dist}")
    print("[PREPROCESSING] Pipeline complete! [OK]")

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_train_resampled': X_train_resampled,
        'y_train_resampled': y_train_resampled,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoders': encoders,
        'pre_smote_distribution': pre_smote_dist,
        'post_smote_distribution': post_smote_dist,
        'X_train_unscaled': X_train,
        'X_test_unscaled': X_test,
    }


def preprocess_single_transaction(txn: dict, scaler, encoders, feature_names) -> pd.DataFrame:
    """
    Preprocess a single transaction for live prediction.

    Parameters
    ----------
    txn : dict
        Transaction data dictionary.
    scaler : RobustScaler
        Fitted scaler from training pipeline.
    encoders : dict
        Fitted label encoders from training pipeline.
    feature_names : list
        Expected feature names in order.

    Returns
    -------
    pd.DataFrame
        Preprocessed single-row DataFrame ready for prediction.
    """
    df = pd.DataFrame([txn])
    df = engineer_features(df)
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    # Select and order features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale numerical columns
    num_cols = [c for c in NUMERICAL_COLS if c in feature_names]
    engineered_num = ['amount_x_distance', 'amount_x_night', 'velocity_x_deviation', 'log_amount']
    num_cols += [c for c in engineered_num if c in feature_names]

    df[num_cols] = scaler.transform(df[num_cols])

    return df
