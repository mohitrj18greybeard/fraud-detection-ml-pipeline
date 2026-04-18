"""
Model Training Module
=====================
Trains and compares Logistic Regression (baseline) and XGBoost (primary)
models for fraud detection. Includes hyperparameter tuning, model saving,
and threshold optimization.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')


def train_logistic_regression(X_train, y_train, random_state=42) -> dict:
    """
    Train a Logistic Regression model with class weight balancing.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features (SMOTE-resampled).
    y_train : pd.Series or np.ndarray
        Training labels.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Model, training time, cross-validation scores.
    """
    print("[MODEL] Training Logistic Regression...")
    start = time.time()

    model = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 2)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

    print(f"[MODEL] Logistic Regression trained in {train_time}s")
    print(f"[MODEL] CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        'model': model,
        'name': 'Logistic Regression',
        'train_time': train_time,
        'cv_f1_mean': round(cv_scores.mean(), 4),
        'cv_f1_std': round(cv_scores.std(), 4),
    }


def train_xgboost(X_train, y_train, X_val=None, y_val=None,
                  scale_pos_weight=None, random_state=42) -> dict:
    """
    Train an XGBoost model optimized for fraud detection.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features (SMOTE-resampled).
    y_train : pd.Series or np.ndarray
        Training labels.
    X_val : pd.DataFrame or np.ndarray, optional
        Validation features for early stopping.
    y_val : pd.Series or np.ndarray, optional
        Validation labels.
    scale_pos_weight : float, optional
        Ratio of negative to positive class. Auto-calculated if None.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Model, training time, feature importances, cross-validation scores.
    """
    print("[MODEL] Training XGBoost...")
    start = time.time()

    if scale_pos_weight is None:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
        tree_method='hist',
    )

    # Train with early stopping if validation set provided
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    train_time = round(time.time() - start, 2)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

    print(f"[MODEL] XGBoost trained in {train_time}s")
    print(f"[MODEL] CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        'model': model,
        'name': 'XGBoost',
        'train_time': train_time,
        'cv_f1_mean': round(cv_scores.mean(), 4),
        'cv_f1_std': round(cv_scores.std(), 4),
    }


def find_optimal_threshold(model, X_test, y_test) -> dict:
    """
    Find the optimal classification threshold that maximizes F1-score.

    Parameters
    ----------
    model : trained model
        Model with predict_proba method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    dict
        Optimal threshold and corresponding metrics.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)

    best_f1 = 0
    best_threshold = 0.5
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        results.append({'threshold': round(t, 2), 'f1': f1, 'precision': precision, 'recall': recall})

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(t, 2)

    results_df = pd.DataFrame(results)

    return {
        'optimal_threshold': best_threshold,
        'best_f1': round(best_f1, 4),
        'threshold_results': results_df,
    }


def save_models(lr_result: dict, xgb_result: dict,
                scaler, encoders, feature_names: list,
                save_dir: str = None) -> str:
    """
    Save trained models and preprocessing artifacts to disk.

    Parameters
    ----------
    lr_result : dict
        Logistic regression training result.
    xgb_result : dict
        XGBoost training result.
    scaler : RobustScaler
        Fitted scaler.
    encoders : dict
        Fitted label encoders.
    feature_names : list
        Feature column names.
    save_dir : str, optional
        Directory to save models.

    Returns
    -------
    str
        Path to saved models directory.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(lr_result['model'], os.path.join(save_dir, 'logistic_regression.pkl'))
    joblib.dump(xgb_result['model'], os.path.join(save_dir, 'xgboost_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))

    # Save model metadata
    metadata = {
        'lr_cv_f1': lr_result['cv_f1_mean'],
        'lr_train_time': lr_result['train_time'],
        'xgb_cv_f1': xgb_result['cv_f1_mean'],
        'xgb_train_time': xgb_result['train_time'],
        'n_features': len(feature_names),
        'feature_names': feature_names,
    }
    joblib.dump(metadata, os.path.join(save_dir, 'metadata.pkl'))

    print(f"[MODEL] All models saved to {save_dir}")
    return save_dir


def load_models(models_dir: str = None) -> dict:
    """
    Load trained models and preprocessing artifacts from disk.

    Parameters
    ----------
    models_dir : str, optional
        Directory containing saved models.

    Returns
    -------
    dict
        Dictionary containing all loaded artifacts.
    """
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

    return {
        'lr_model': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
        'xgb_model': joblib.load(os.path.join(models_dir, 'xgboost_model.pkl')),
        'scaler': joblib.load(os.path.join(models_dir, 'scaler.pkl')),
        'encoders': joblib.load(os.path.join(models_dir, 'encoders.pkl')),
        'feature_names': joblib.load(os.path.join(models_dir, 'feature_names.pkl')),
        'metadata': joblib.load(os.path.join(models_dir, 'metadata.pkl')),
    }
