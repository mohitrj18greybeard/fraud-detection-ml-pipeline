"""
FraudShield AI - Evaluation Module.
Handles metrics calculation and visualization for model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, auc, accuracy_score
)

def evaluate_model(model, X_test, y_test, name, threshold=0.5):
    """Calculate comprehensive evaluation metrics."""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate Savings (Simplified Business Impact)
    # Assume we save 100% of caught fraud amount, but each false positive costs $50 in manual review
    avg_fraud_val = 250 # Estimated
    false_positives = cm[0, 1]
    true_positives = cm[1, 1]
    potential_savings = (true_positives * avg_fraud_val) - (false_positives * 50)
    
    return {
        'name': name,
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_fraud': report['1']['precision'],
        'recall_fraud': report['1']['recall'],
        'f1_fraud': report['1']['f1-score'],
        'roc_auc': roc_auc_score(y_test, y_probs),
        'pr_auc': auc(precision_recall_curve(y_test, y_probs)[1], precision_recall_curve(y_test, y_probs)[0]),
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_probs': y_probs,
        'threshold': threshold,
        'potential_savings': potential_savings
    }

def plot_confusion_matrix(eval_result):
    """Plot confusion matrix with labels."""
    cm = eval_result['confusion_matrix']
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {eval_result["name"]}', color='white', fontweight='bold')
    ax.set_xlabel('Predicted Label', color='white')
    ax.set_ylabel('True Label', color='white')
    ax.set_xticklabels(['Legit', 'Fraud'], color='white')
    ax.set_yticklabels(['Legit', 'Fraud'], color='white')
    plt.tight_layout()
    return fig

def plot_roc_curves(eval_results):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    for res in eval_results:
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_probs'])
        ax.plot(fpr, tpr, label=f'{res["name"]} (AUC = {res["roc_auc"]:.3f})', linewidth=2)
        
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', color='white', fontweight='bold')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.legend(facecolor='#1E2130', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    return fig

def plot_precision_recall_curves(eval_results):
    """Plot PR curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    for res in eval_results:
        precision, recall, _ = precision_recall_curve(res['y_test'], res['y_probs'])
        ax.plot(recall, precision, label=f'{res["name"]} (AUC = {res["pr_auc"]:.3f})', linewidth=2)
        
    ax.set_title('Precision-Recall Curve', color='white', fontweight='bold')
    ax.set_xlabel('Recall', color='white')
    ax.set_ylabel('Precision', color='white')
    ax.legend(facecolor='#1E2130', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    return fig

def plot_threshold_analysis(results_df, model_name):
    """Plot F1, Precision, and Recall vs Threshold."""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.plot(results_df['threshold'], results_df['precision'], label='Precision', linestyle='--', color='#667EEA')
    ax.plot(results_df['threshold'], results_df['recall'], label='Recall', linestyle='--', color='#00D4AA')
    ax.plot(results_df['threshold'], results_df['f1'], label='F1-Score', linewidth=3, color='#FF6B6B')
    
    ax.set_title(f'Threshold Analysis - {model_name}', color='white', fontweight='bold')
    ax.set_xlabel('Decision Threshold', color='white')
    ax.set_ylabel('Score', color='white')
    ax.legend(facecolor='#1E2130', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    return fig

def plot_model_comparison_bar(eval_results):
    """Plot bar chart comparing key metrics across models."""
    metrics = ['Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'ROC AUC']
    data = []
    for res in eval_results:
        data.append({
            'Model': res['name'],
            'Precision (Fraud)': res['precision_fraud'],
            'Recall (Fraud)': res['recall_fraud'],
            'F1-Score (Fraud)': res['f1_fraud'],
            'ROC AUC': res['roc_auc']
        })
    
    df = pd.DataFrame(data).melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette='viridis', ax=ax)
    ax.set_title('Model Performance Comparison', color='white', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='white')
    ax.set_xlabel('', color='white')
    ax.set_ylabel('Score', color='white')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#1E2130', labelcolor='white')
    plt.tight_layout()
    return fig

def generate_comparison_table(eval_results):
    """Generate a summary table for the dashboard."""
    rows = []
    for res in eval_results:
        rows.append({
            'Model': res['name'],
            'Accuracy': f"{res['accuracy']:.4f}",
            'Precision': f"{res['precision_fraud']:.4f}",
            'Recall': f"{res['recall_fraud']:.4f}",
            'F1-Score': f"{res['f1_fraud']:.4f}",
            'ROC AUC': f"{res['roc_auc']:.4f}",
            'Threshold': f"{res['threshold']:.2f}"
        })
    return pd.DataFrame(rows)
