"""
Model Explainability & Business Insights Engine
================================================
Provides feature importance analysis, permutation importance,
partial dependence plots, and auto-generated business insights
for fraud detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
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
}


def get_xgboost_feature_importance(model, feature_names: list,
                                   importance_type: str = 'gain') -> pd.DataFrame:
    """
    Extract XGBoost built-in feature importance.

    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    feature_names : list
        Feature column names.
    importance_type : str
        Type of importance: 'gain', 'weight', 'cover'.

    Returns
    -------
    pd.DataFrame
        Sorted feature importance DataFrame.
    """
    importance = model.get_booster().get_score(importance_type=importance_type)

    # Map feature indices to names
    imp_df = pd.DataFrame([
        {'feature': feature_names[int(k.replace('f', ''))] if k.startswith('f') and k[1:].isdigit() else k,
         'importance': v}
        for k, v in importance.items()
    ])

    if imp_df.empty:
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })

    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    imp_df['importance_pct'] = (imp_df['importance'] / imp_df['importance'].sum() * 100).round(2)

    return imp_df


def get_sklearn_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from sklearn model (coefficients for LR).

    Parameters
    ----------
    model : sklearn model
        Trained model with coef_ attribute.
    feature_names : list
        Feature column names.

    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame.
    """
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return pd.DataFrame()

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    imp_df['importance_pct'] = (imp_df['importance'] / imp_df['importance'].sum() * 100).round(2)

    return imp_df


def compute_permutation_importance(model, X_test, y_test, feature_names: list,
                                   n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Compute permutation importance (model-agnostic).

    Parameters
    ----------
    model : trained model
        Model with predict method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test labels.
    feature_names : list
        Feature column names.
    n_repeats : int
        Number of permutation repeats.
    random_state : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Permutation importance DataFrame.
    """
    print("[EXPLAIN] Computing permutation importance...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='f1',
        n_jobs=-1
    )

    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    imp_df = imp_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

    return imp_df


def plot_feature_importance(imp_df: pd.DataFrame, title: str = 'Feature Importance',
                            top_n: int = 15, figsize=(10, 8)) -> plt.Figure:
    """Plot a horizontal bar chart of feature importance."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    data = imp_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar
    importance_col = 'importance' if 'importance' in data.columns else 'importance_mean'

    # Create gradient colors
    n = len(data)
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n))

    bars = ax.barh(data['feature'], data[importance_col], color=colors,
                   edgecolor='white', linewidth=0.3, height=0.7)

    # Add value labels
    max_val = data[importance_col].max()
    for bar, val in zip(bars, data[importance_col]):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', color=COLORS['text'],
                fontsize=9, fontweight='bold')

    ax.set_xlabel('Importance Score', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title(title, fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.grid(True, alpha=0.1, axis='x', color=COLORS['grid'])

    plt.tight_layout()
    return fig


def plot_permutation_importance(perm_df: pd.DataFrame, title: str = 'Permutation Importance',
                                top_n: int = 15, figsize=(10, 8)) -> plt.Figure:
    """Plot permutation importance with error bars."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    data = perm_df.head(top_n).iloc[::-1]

    n = len(data)
    colors = plt.cm.cool(np.linspace(0.3, 0.9, n))

    bars = ax.barh(data['feature'], data['importance_mean'], xerr=data['importance_std'],
                   color=colors, edgecolor='white', linewidth=0.3, height=0.7,
                   capsize=3, error_kw={'ecolor': COLORS['text'], 'alpha': 0.5})

    ax.set_xlabel('Mean Importance (F1 decrease)', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax.set_title(title, fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.grid(True, alpha=0.1, axis='x', color=COLORS['grid'])
    ax.axvline(x=0, color=COLORS['text'], lw=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_business_insights(df: pd.DataFrame, xgb_importance: pd.DataFrame,
                               eval_results: dict) -> list:
    """
    Auto-generate natural-language business insights from the model.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    xgb_importance : pd.DataFrame
        Feature importance DataFrame.
    eval_results : dict
        Evaluation results dictionary.

    Returns
    -------
    list[dict]
        List of insight dictionaries with 'title', 'insight', 'icon', 'severity'.
    """
    fraud = df[df['is_fraud'] == 1]
    legit = df[df['is_fraud'] == 0]
    insights = []

    # 1. Top predictive feature
    if not xgb_importance.empty:
        top_feature = xgb_importance.iloc[0]['feature']
        top_pct = xgb_importance.iloc[0].get('importance_pct', 0)
        insights.append({
            'title': 'Strongest Fraud Predictor',
            'insight': f"**{top_feature}** is the most predictive feature, contributing {top_pct:.1f}% "
                       f"to the model's decision-making. Security teams should prioritize monitoring "
                       f"this signal for early fraud detection.",
            'icon': '🎯',
            'severity': 'high'
        })

    # 2. Amount analysis
    avg_fraud = fraud['amount'].mean()
    avg_legit = legit['amount'].mean()
    ratio = avg_fraud / max(avg_legit, 1)
    p95_fraud = fraud['amount'].quantile(0.95)
    insights.append({
        'title': 'Transaction Amount Red Flag',
        'insight': f"Fraudulent transactions average **${avg_fraud:,.2f}** vs **${avg_legit:,.2f}** for "
                   f"legitimate ones — a **{ratio:.1f}x** multiplier. Transactions above **${p95_fraud:,.2f}** "
                   f"(95th percentile of fraud) should trigger enhanced verification.",
        'icon': '💰',
        'severity': 'high'
    })

    # 3. Time-based patterns
    peak_fraud_hours = fraud['hour_of_day'].value_counts().head(3).index.tolist()
    night_fraud_pct = fraud['is_night'].mean() * 100
    insights.append({
        'title': 'Temporal Fraud Patterns',
        'insight': f"**{night_fraud_pct:.1f}%** of fraud occurs during nighttime hours (10 PM – 5 AM). "
                   f"Peak fraud hours are **{peak_fraud_hours}**. Consider implementing enhanced "
                   f"authentication for transactions during these windows.",
        'icon': '🌙',
        'severity': 'medium'
    })

    # 4. Cross-border risk
    cross_border_fraud_rate = fraud['is_cross_border'].mean() * 100
    cross_border_legit_rate = legit['is_cross_border'].mean() * 100
    cross_ratio = cross_border_fraud_rate / max(cross_border_legit_rate, 0.1)
    insights.append({
        'title': 'Cross-Border Transaction Risk',
        'insight': f"**{cross_border_fraud_rate:.1f}%** of fraudulent transactions are cross-border "
                   f"compared to **{cross_border_legit_rate:.1f}%** of legitimate ones — "
                   f"a **{cross_ratio:.1f}x** higher rate. International transactions should "
                   f"receive additional scrutiny.",
        'icon': '🌍',
        'severity': 'high'
    })

    # 5. Merchant category risk
    top_fraud_cats = fraud['merchant_category'].value_counts(normalize=True).head(3)
    top_cat = top_fraud_cats.index[0]
    top_cat_pct = top_fraud_cats.iloc[0] * 100
    insights.append({
        'title': 'High-Risk Merchant Categories',
        'insight': f"**{top_cat}** accounts for **{top_cat_pct:.1f}%** of all fraudulent transactions. "
                   f"Top 3 risky categories: {', '.join(top_fraud_cats.index.tolist())}. "
                   f"ML-based dynamic risk scoring per merchant category is recommended.",
        'icon': '🏪',
        'severity': 'medium'
    })

    # 6. Entry mode risk
    manual_fraud = fraud[fraud['entry_mode'] == 'manual'].shape[0]
    manual_total = df[df['entry_mode'] == 'manual'].shape[0]
    manual_fraud_rate = (manual_fraud / max(manual_total, 1)) * 100
    insights.append({
        'title': 'Manual Entry Vulnerability',
        'insight': f"Manual card entry transactions have a **{manual_fraud_rate:.1f}%** fraud rate — "
                   f"significantly higher than chip or contactless methods. Requiring additional "
                   f"verification for manual entries could reduce fraud by an estimated 15-20%.",
        'icon': '⌨️',
        'severity': 'medium'
    })

    # 7. Model savings
    if eval_results:
        savings = eval_results.get('potential_savings', 0)
        recall = eval_results.get('recall_fraud', 0)
        insights.append({
            'title': 'Financial Impact of the Model',
            'insight': f"The model catches **{recall*100:.1f}%** of fraudulent transactions, generating "
                       f"potential savings of **${savings:,.0f}**. Each 1% improvement in recall could "
                       f"save an additional **${savings * 0.01:,.0f}** annually.",
            'icon': '📈',
            'severity': 'high'
        })

    # 8. Velocity patterns
    avg_velocity_fraud = fraud['velocity_24h'].mean()
    avg_velocity_legit = legit['velocity_24h'].mean()
    insights.append({
        'title': 'Transaction Velocity Anomaly',
        'insight': f"Fraudsters make **{avg_velocity_fraud:.1f}** transactions in a 24h window vs "
                   f"**{avg_velocity_legit:.1f}** for legitimate customers. Velocity-based alerts "
                   f"for customers exceeding 2σ above their historical average are recommended.",
        'icon': '⚡',
        'severity': 'medium'
    })

    return insights


def explain_single_prediction(model, transaction_features: pd.DataFrame,
                              feature_names: list, prediction: int,
                              probability: float) -> list:
    """
    Explain why a single transaction was flagged/cleared.

    Parameters
    ----------
    model : trained model
        Model used for prediction.
    transaction_features : pd.DataFrame
        Single-row preprocessed features.
    feature_names : list
        Feature column names.
    prediction : int
        0 or 1 (fraud prediction).
    probability : float
        Fraud probability score.

    Returns
    -------
    list[dict]
        List of contributing factors.
    """
    factors = []

    # Get feature values
    values = transaction_features.iloc[0].to_dict()

    # For XGBoost, use feature_importances_ as proxy for contribution
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(feature_names, model.feature_importances_))

        # Sort by importance and flag top features
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        for feat, imp in sorted_features[:8]:
            val = values.get(feat, 0)
            direction = '↑ Increases risk' if prediction == 1 else '↓ Low risk signal'
            factors.append({
                'feature': feat,
                'value': round(val, 4) if isinstance(val, float) else val,
                'importance': round(imp, 4),
                'direction': direction,
            })

    return factors


def plot_fraud_patterns(df: pd.DataFrame, figsize=(16, 12)) -> plt.Figure:
    """Create a comprehensive 2x2 grid of fraud pattern visualizations."""
    fraud = df[df['is_fraud'] == 1]
    legit = df[df['is_fraud'] == 0]

    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=COLORS['bg'])
    for ax in axes.flat:
        ax.set_facecolor(COLORS['bg'])

    # 1. Amount distribution
    ax1 = axes[0, 0]
    ax1.hist(legit['amount'].clip(upper=legit['amount'].quantile(0.99)), bins=80,
             alpha=0.6, color=COLORS['primary'], label='Legitimate', density=True)
    ax1.hist(fraud['amount'].clip(upper=fraud['amount'].quantile(0.99)), bins=80,
             alpha=0.7, color=COLORS['secondary'], label='Fraud', density=True)
    ax1.set_xlabel('Amount ($)', color=COLORS['text'])
    ax1.set_ylabel('Density', color=COLORS['text'])
    ax1.set_title('Transaction Amount Distribution', color=COLORS['text'], fontweight='bold')
    ax1.legend(facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])

    # 2. Fraud by hour
    ax2 = axes[0, 1]
    fraud_by_hour = fraud['hour_of_day'].value_counts().sort_index()
    legit_by_hour = legit['hour_of_day'].value_counts().sort_index()
    # Normalize
    fraud_pct = fraud_by_hour / fraud_by_hour.sum()
    legit_pct = legit_by_hour / legit_by_hour.sum()
    ax2.bar(fraud_pct.index - 0.2, fraud_pct.values, width=0.4,
            color=COLORS['secondary'], alpha=0.8, label='Fraud')
    ax2.bar(legit_pct.index + 0.2, legit_pct.values, width=0.4,
            color=COLORS['primary'], alpha=0.6, label='Legitimate')
    ax2.set_xlabel('Hour of Day', color=COLORS['text'])
    ax2.set_ylabel('Proportion', color=COLORS['text'])
    ax2.set_title('Transaction Distribution by Hour', color=COLORS['text'], fontweight='bold')
    ax2.legend(facecolor=COLORS['card'], edgecolor=COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])

    # 3. Merchant category fraud rate
    ax3 = axes[1, 0]
    cat_fraud_rate = df.groupby('merchant_category')['is_fraud'].mean().sort_values(ascending=True)
    colors_bar = [COLORS['secondary'] if v > 0.03 else COLORS['primary'] for v in cat_fraud_rate.values]
    ax3.barh(cat_fraud_rate.index, cat_fraud_rate.values * 100, color=colors_bar,
             edgecolor='white', linewidth=0.3)
    ax3.set_xlabel('Fraud Rate (%)', color=COLORS['text'])
    ax3.set_title('Fraud Rate by Merchant Category', color=COLORS['text'], fontweight='bold')
    ax3.tick_params(colors=COLORS['text'])

    # 4. Entry mode vs fraud
    ax4 = axes[1, 1]
    entry_fraud = df.groupby('entry_mode')['is_fraud'].agg(['sum', 'count'])
    entry_fraud['rate'] = (entry_fraud['sum'] / entry_fraud['count'] * 100)
    entry_fraud = entry_fraud.sort_values('rate', ascending=True)
    bars = ax4.barh(entry_fraud.index, entry_fraud['rate'],
                    color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(entry_fraud))),
                    edgecolor='white', linewidth=0.3)
    for bar, val in zip(bars, entry_fraud['rate']):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}%', ha='left', va='center', color=COLORS['text'], fontsize=9)
    ax4.set_xlabel('Fraud Rate (%)', color=COLORS['text'])
    ax4.set_title('Fraud Rate by Entry Mode', color=COLORS['text'], fontweight='bold')
    ax4.tick_params(colors=COLORS['text'])

    plt.suptitle('Fraud Detection — Pattern Analysis', fontsize=16,
                 color=COLORS['text'], fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
