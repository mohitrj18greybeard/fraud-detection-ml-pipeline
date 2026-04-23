"""
FraudShield AI - Modeling Module.
Handles model training (LR/XGB) and threshold optimization.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve

def train_logistic_regression(X_train, y_train):
    """Train a baseline Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return {'model': model, 'name': 'Logistic Regression'}

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train a high-performance XGBoost model."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return {'model': model, 'name': 'XGBoost'}

def find_optimal_threshold(model, X_test, y_test):
    """Find the threshold that maximizes F1-score for the fraud class."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    results = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions[:-1],
        'recall': recalls[:-1],
        'f1': f1_scores[:-1]
    })
    
    return {
        'optimal_threshold': best_threshold,
        'best_f1': f1_scores[best_idx],
        'threshold_results': results
    }

def save_models(models_dict, base_path="models"):
    """Persist models to disk."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for name, model in models_dict.items():
        path = os.path.join(base_path, f"{name}.pkl")
        joblib.dump(model, path)
        
def load_models(base_path="models"):
    """Load models from disk."""
    models = {}
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                models[name] = joblib.load(os.path.join(base_path, file))
    return models
