"""
Model Evaluation Engine
=======================
Comprehensive evaluation suite for fraud detection models.
Generates classification reports, confusion matrices, ROC curves,
precision-recall curves, and cost-benefit analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef
)
import warnings
warnings.filterwarnings('ignore')

# --- Styling ---
plt.style.use('dark_background')
COLORS = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'accent': '#4ECDC4',
    'warning': '#FFE66D',
    'bg': '#0E1117',
    'card': '#1E2130',
    'text': '#FAFAFA',
    'grid': '#2D3250',
    'gradient_start': '#667EEA',
    'gradient_end': '#764BA2',
}


def evaluate_model(model, X_test, y_test, model_name: str = 'Model',
                   threshold: float = 0.5) -> dict:
    """
    Comprehensive model evaluation.

    Parameters
    ----------
    model : trained sklearn/xgboost model
        Model with predict and predict_proba methods.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.
    model_name : str
        Name for display/logging.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        All evaluation metrics and data.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Core metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    # Additional metrics
    mcc = matthews_corrcoef(y_test, y_pred)

    # Cost analysis (assume: false negative costs $5000, false positive costs $50)
    tn, fp, fn, tp = cm.ravel()
    cost_fn = 5000  # Missing a fraud
    cost_fp = 50    # Flagging legitimate transaction
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    potential_savings = tp * cost_fn  # Caught fraud savings

    results = {
        'model_name': model_name,
        'threshold': threshold,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision_fraud': round(report['1']['precision'], 4),
        'recall_fraud': round(report['1']['recall'], 4),
        'f1_fraud': round(report['1']['f1-score'], 4),
        'precision_legit': round(report['0']['precision'], 4),
        'recall_legit': round(report['0']['recall'], 4),
        'f1_legit': round(report['0']['f1-score'], 4),
        'f1_macro': round(report['macro avg']['f1-score'], 4),
        'f1_weighted': round(report['weighted avg']['f1-score'], 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'mcc': round(mcc, 4),
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'fpr': fpr, 'tpr': tpr,
        'precision_curve': precision_vals, 'recall_curve': recall_vals,
        'y_proba': y_proba,
        'y_pred': y_pred,
        'total_cost': total_cost,
        'potential_savings': potential_savings,
        'cost_per_fn': cost_fn,
        'cost_per_fp': cost_fp,
        'classification_report': report,
    }

    print(f"\n{'='*55}")
    print(f"  {model_name} Evaluation Results (threshold={threshold})")
    print(f"{'='*55}")
    print(f"  Accuracy:           {results['accuracy']:.4f}")
    print(f"  Precision (Fraud):  {results['precision_fraud']:.4f}")
    print(f"  Recall (Fraud):     {results['recall_fraud']:.4f}")
    print(f"  F1-Score (Fraud):   {results['f1_fraud']:.4f}")
    print(f"  ROC AUC:            {results['roc_auc']:.4f}")
    print(f"  PR AUC:             {results['pr_auc']:.4f}")
    print(f"  MCC:                {results['mcc']:.4f}")
    print(f"  Potential Savings:  ${results['potential_savings']:,.0f}")
    print(f"{'='*55}")

    return results


def plot_confusion_matrix(results: dict, figsize=(8, 6)) -> plt.Figure:
    """Plot an elegant confusion matrix heatmap."""
    cm = results['confusion_matrix']
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # Create annotation labels with counts and percentages
    labels = np.array([
        [f'TN\n{cm[0,0]:,}\n({cm_pct[0,0]:.1f}%)', f'FP\n{cm[0,1]:,}\n({cm_pct[0,1]:.1f}%)'],
        [f'FN\n{cm[1,0]:,}\n({cm_pct[1,0]:.1f}%)', f'TP\n{cm[1,1]:,}\n({cm_pct[1,1]:.1f}%)']
    ])

    sns.heatmap(cm, annot=labels, fmt='', cmap='YlOrRd',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                ax=ax, linewidths=2, linecolor=COLORS['bg'],
                cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title(f"{results['model_name']} — Confusion Matrix",
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    return fig


def plot_roc_curves(results_list: list, figsize=(10, 7)) -> plt.Figure:
    """Plot ROC curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']]

    for i, results in enumerate(results_list):
        color = colors[i % len(colors)]
        ax.plot(results['fpr'], results['tpr'],
                color=color, lw=2.5,
                label=f"{results['model_name']} (AUC = {results['roc_auc']:.4f})")

    ax.plot([0, 1], [0, 1], 'w--', lw=1, alpha=0.3, label='Random Classifier')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='white')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    return fig


def plot_precision_recall_curves(results_list: list, figsize=(10, 7)) -> plt.Figure:
    """Plot Precision-Recall curves (more informative than ROC for imbalanced data)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning']]

    for i, results in enumerate(results_list):
        color = colors[i % len(colors)]
        ax.plot(results['recall_curve'], results['precision_curve'],
                color=color, lw=2.5,
                label=f"{results['model_name']} (AP = {results['pr_auc']:.4f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('Recall', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison (Key for Imbalanced Data)',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    return fig


def plot_threshold_analysis(threshold_results: pd.DataFrame, model_name: str,
                            figsize=(12, 6)) -> plt.Figure:
    """Plot how metrics change with classification threshold."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    ax.plot(threshold_results['threshold'], threshold_results['f1'],
            color=COLORS['primary'], lw=2.5, label='F1-Score')
    ax.plot(threshold_results['threshold'], threshold_results['precision'],
            color=COLORS['secondary'], lw=2, label='Precision', linestyle='--')
    ax.plot(threshold_results['threshold'], threshold_results['recall'],
            color=COLORS['accent'], lw=2, label='Recall', linestyle='-.')

    # Mark optimal threshold
    best_idx = threshold_results['f1'].idxmax()
    best_t = threshold_results.loc[best_idx, 'threshold']
    best_f1 = threshold_results.loc[best_idx, 'f1']
    ax.axvline(x=best_t, color=COLORS['warning'], lw=1.5, linestyle=':', alpha=0.8)
    ax.scatter([best_t], [best_f1], color=COLORS['warning'], s=100, zorder=5,
               label=f'Optimal (t={best_t:.2f}, F1={best_f1:.3f})')

    ax.set_xlabel('Classification Threshold', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title(f'{model_name} — Threshold Analysis',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    return fig


def plot_model_comparison_bar(results_list: list, figsize=(12, 7)) -> plt.Figure:
    """Create a grouped bar chart comparing key metrics across models."""
    metrics = ['precision_fraud', 'recall_fraud', 'f1_fraud', 'roc_auc', 'pr_auc', 'mcc']
    metric_labels = ['Precision\n(Fraud)', 'Recall\n(Fraud)', 'F1-Score\n(Fraud)',
                     'ROC AUC', 'PR AUC', 'MCC']

    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(metrics))
    width = 0.35
    colors = [COLORS['primary'], COLORS['secondary']]

    for i, results in enumerate(results_list):
        values = [results[m] for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset + width/2, values, width, label=results['model_name'],
                      color=colors[i % len(colors)], alpha=0.85, edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color=COLORS['text'],
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title('Model Comparison — Key Metrics',
                 fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=COLORS['text'])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax.grid(True, alpha=0.1, axis='y', color=COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    return fig


def generate_comparison_table(results_list: list) -> pd.DataFrame:
    """Generate a formatted comparison table of model metrics."""
    rows = []
    for r in results_list:
        rows.append({
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'Precision (Fraud)': r['precision_fraud'],
            'Recall (Fraud)': r['recall_fraud'],
            'F1 (Fraud)': r['f1_fraud'],
            'ROC AUC': r['roc_auc'],
            'PR AUC': r['pr_auc'],
            'MCC': r['mcc'],
            'TP': r['tp'],
            'FP': r['fp'],
            'FN': r['fn'],
            'TN': r['tn'],
            'Potential Savings ($)': f"${r['potential_savings']:,.0f}",
            'Total Cost ($)': f"${r['total_cost']:,.0f}",
        })
    return pd.DataFrame(rows)
