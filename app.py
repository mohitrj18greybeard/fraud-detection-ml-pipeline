"""
AI-Powered Financial Fraud Detection System
============================================
Elite-level Streamlit dashboard for real-time fraud detection,
model comparison, explainability, and business intelligence.

Author: Mohit
"""

import os
import sys
import joblib

# model_path = os.path.join(base_dir, "models", "model.pkl")
# model = joblib.load(model_path)

# Fix encoding for cloud environments
os.environ['PYTHONIOENCODING'] = 'utf-8'

import streamlit as st

# @st.cache_resource
# def load_model():
#     return joblib.load("models/model.pkl")
# 
# model = load_model()

# PAGE CONFIG MUST be the FIRST Streamlit command (before any other imports
# that might trigger Streamlit internally)
st.set_page_config(
    page_title="FraudShield AI -- Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_dataset, get_dataset_summary, MERCHANT_CATEGORIES, CARD_TYPES, ENTRY_MODES, COUNTRIES
from src.preprocessing import preprocess_pipeline, preprocess_single_transaction
from src.model import train_logistic_regression, train_xgboost, find_optimal_threshold, save_models, load_models
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves, plot_threshold_analysis,
    plot_model_comparison_bar, generate_comparison_table
)
from src.explainability import (
    get_xgboost_feature_importance, get_sklearn_feature_importance,
    compute_permutation_importance, plot_feature_importance,
    plot_permutation_importance, generate_business_insights,
    explain_single_prediction, plot_fraud_patterns
)

# ╔══════════════════════════════════════════════════════════════════╗
# ║                     CUSTOM CSS                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ===== HEADER ===== */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0,212,170,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00D4AA, #4ECDC4, #667EEA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #a0a0b0;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* ===== KPI CARDS ===== */
    .kpi-card {
        background: linear-gradient(145deg, #1E2130, #252940);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0 0.25rem 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8890a4;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .kpi-green { color: #00D4AA; }
    .kpi-red { color: #FF6B6B; }
    .kpi-blue { color: #667EEA; }
    .kpi-yellow { color: #FFE66D; }

    /* ===== INSIGHT CARDS ===== */
    .insight-card {
        background: linear-gradient(145deg, #1a1d2e, #22263a);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00D4AA;
        transition: border-color 0.2s ease;
    }
    .insight-card:hover {
        border-left-color: #667EEA;
    }
    .insight-card.high {
        border-left-color: #FF6B6B;
    }
    .insight-card.medium {
        border-left-color: #FFE66D;
    }
    .insight-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #FAFAFA;
    }
    .insight-text {
        font-size: 0.9rem;
        color: #b0b8c8;
        line-height: 1.6;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FAFAFA;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0,212,170,0.3);
    }

    /* ===== BADGES ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-fraud { background: rgba(255,107,107,0.2); color: #FF6B6B; }
    .badge-legit { background: rgba(0,212,170,0.2); color: #00D4AA; }
    .badge-warning { background: rgba(255,230,109,0.2); color: #FFE66D; }

    /* ===== RISK METER ===== */
    .risk-meter {
        background: linear-gradient(90deg, #00D4AA 0%, #FFE66D 50%, #FF6B6B 100%);
        height: 10px;
        border-radius: 5px;
        position: relative;
        margin: 1rem 0;
    }
    .risk-indicator {
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 50%;
        position: absolute;
        top: -5px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* ===== PIPELINE STEP ===== */
    .pipeline-step {
        background: linear-gradient(145deg, #1a1d2e, #22263a);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.05);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .pipeline-number {
        background: linear-gradient(135deg, #00D4AA, #4ECDC4);
        color: #0E1117;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        flex-shrink: 0;
    }

    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {visibility: hidden;}

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1d2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(90deg, #00D4AA, #667EEA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     CACHING & DATA LOADING                       ║
# ╚══════════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False)
def load_or_generate_data():
    """Load existing data or generate new synthetic dataset."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'transactions.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df = generate_dataset(save_path=data_path)
    return df


@st.cache_data(show_spinner=False)
def run_preprocessing(df):
    """Run the full preprocessing pipeline."""
    return preprocess_pipeline(df)


@st.cache_resource(show_spinner=False)
def train_models_cached(X_train_resampled, y_train_resampled, X_test, y_test):
    """Train both models (cached)."""
    lr_result = train_logistic_regression(X_train_resampled, y_train_resampled)
    xgb_result = train_xgboost(X_train_resampled, y_train_resampled, X_test, y_test)
    return lr_result, xgb_result


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     SIDEBAR NAVIGATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown("# 🛡️ FraudShield AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Summary",
            "📊 Data Exploration",
            "⚙️ ML Pipeline",
            "🤖 Model Performance",
            "🔍 Explainability & Insights",
            "🎯 Live Prediction"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c7293; font-size: 0.75rem;'>
        <p>Built with ❤️ by Mohit RJ</p>
        <p>Powered by XGBoost + SMOTE</p>
        <p style='margin-top: 0.5rem;'>v2.0 — Production Grade</p>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     INITIALIZE DATA & MODELS                     ║
# ╚══════════════════════════════════════════════════════════════════╝
with st.spinner("🔄 Initializing FraudShield AI engine..."):
    df = load_or_generate_data()
    pipeline = run_preprocessing(df)
    lr_result, xgb_result = train_models_cached(
        pipeline['X_train_resampled'],
        pipeline['y_train_resampled'],
        pipeline['X_test'],
        pipeline['y_test']
    )

# Evaluate both models
lr_eval = evaluate_model(lr_result['model'], pipeline['X_test'], pipeline['y_test'], 'Logistic Regression')
xgb_eval = evaluate_model(xgb_result['model'], pipeline['X_test'], pipeline['y_test'], 'XGBoost')
summary = get_dataset_summary(df)

# Threshold optimization for XGBoost
xgb_threshold = find_optimal_threshold(xgb_result['model'], pipeline['X_test'], pipeline['y_test'])
xgb_eval_optimal = evaluate_model(
    xgb_result['model'], pipeline['X_test'], pipeline['y_test'],
    'XGBoost (Optimized)', threshold=xgb_threshold['optimal_threshold']
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: EXECUTIVE SUMMARY                      ║
# ╚══════════════════════════════════════════════════════════════════╝
if page == "🏠 Executive Summary":
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🛡️ FraudShield AI — Executive Dashboard</div>
        <div class="hero-subtitle">
            Real-time AI-powered financial fraud detection system • Protecting $%.1fM in transactions
        </div>
    </div>
    """ % (summary['total_amount'] / 1_000_000), unsafe_allow_html=True)

    # KPI Row 1
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value kpi-blue">{summary['total_transactions']:,}</div>
            <div class="kpi-delta">6-month analysis period</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Fraud Detected</div>
            <div class="kpi-value kpi-red">{summary['total_fraud']:,}</div>
            <div class="kpi-delta">{summary['fraud_rate']:.2f}% fraud rate</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">XGBoost F1 Score</div>
            <div class="kpi-value kpi-green">{xgb_eval['f1_fraud']:.4f}</div>
            <div class="kpi-delta">Fraud class F1</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Potential Savings</div>
            <div class="kpi-value kpi-yellow">${xgb_eval['potential_savings']:,.0f}</div>
            <div class="kpi-delta">From caught fraud</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # KPI Row 2
    k5, k6, k7, k8 = st.columns(4)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Recall (Fraud)</div>
            <div class="kpi-value kpi-green">{xgb_eval['recall_fraud']*100:.1f}%</div>
            <div class="kpi-delta">% of fraud caught</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Precision (Fraud)</div>
            <div class="kpi-value kpi-blue">{xgb_eval['precision_fraud']*100:.1f}%</div>
            <div class="kpi-delta">% of alerts that are real</div>
        </div>
        """, unsafe_allow_html=True)
    with k7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">ROC AUC</div>
            <div class="kpi-value kpi-green">{xgb_eval['roc_auc']:.4f}</div>
            <div class="kpi-delta">Area under ROC curve</div>
        </div>
        """, unsafe_allow_html=True)
    with k8:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Fraud Amount</div>
            <div class="kpi-value kpi-red">${summary['avg_fraud_amount']:,.0f}</div>
            <div class="kpi-delta">vs ${summary['avg_legit_amount']:,.0f} legit</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Quick Model Comparison
    st.markdown('<div class="section-header">📊 Model Comparison at a Glance</div>', unsafe_allow_html=True)
    comparison_df = generate_comparison_table([lr_eval, xgb_eval])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 Model Comparison — Key Metrics**")
        fig = plot_model_comparison_bar([lr_eval, xgb_eval])
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**📈 ROC Curve Comparison**")
        fig = plot_roc_curves([lr_eval, xgb_eval])
        st.pyplot(fig)
        plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: DATA EXPLORATION                       ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "📊 Data Exploration":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📊 Data Exploration & Analysis</div>
        <div class="hero-subtitle">
            Deep-dive into %s transactions • Understanding fraud patterns and distributions
        </div>
    </div>
    """ % f"{summary['total_transactions']:,}", unsafe_allow_html=True)

    # Dataset Overview
    st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Features", f"{len(df.columns) - 1}")
        st.metric("Unique Customers", f"{summary['n_customers']:,}")
    with col2:
        st.metric("Fraud Transactions", f"{summary['total_fraud']:,}")
        st.metric("Legitimate Transactions", f"{summary['total_legitimate']:,}")
        st.metric("Imbalance Ratio", f"1:{summary['total_legitimate']//max(summary['total_fraud'],1)}")
    with col3:
        st.metric("Total $ Volume", f"${summary['total_amount']:,.0f}")
        st.metric("Fraud $ Volume", f"${summary['fraud_amount']:,.0f}")
        st.metric("Date Range", f"{summary['date_range'][0].strftime('%b %d')} — {summary['date_range'][1].strftime('%b %d, %Y')}")

    # Class Distribution
    st.markdown('<div class="section-header">⚖️ Class Imbalance Visualization</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        counts = df['is_fraud'].value_counts()
        bars = ax.bar(['Legitimate', 'Fraud'], counts.values,
                      color=['#00D4AA', '#FF6B6B'], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                    f'{val:,}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)
        ax.set_ylabel('Count', color='white', fontsize=12)
        ax.set_title('Transaction Class Distribution', color='white', fontweight='bold', fontsize=14)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=['Legitimate', 'Fraud'],
            colors=['#00D4AA', '#FF6B6B'], autopct='%1.2f%%',
            startangle=90, explode=(0, 0.1),
            textprops={'color': 'white', 'fontsize': 12}
        )
        autotexts[1].set_fontweight('bold')
        ax.set_title('Fraud vs Legitimate Distribution', color='white', fontweight='bold', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Fraud Patterns
    st.markdown('<div class="section-header">🔍 Fraud Pattern Analysis</div>', unsafe_allow_html=True)
    fig = plot_fraud_patterns(df)
    st.pyplot(fig)
    plt.close(fig)

    # Correlation Heatmap
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5, linecolor='#1E2130',
                cbar_kws={'label': 'Correlation'}, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Matrix', color='white', fontweight='bold', fontsize=14, pad=15)
    ax.tick_params(colors='white', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Data Sample
    st.markdown('<div class="section-header">📄 Sample Data</div>', unsafe_allow_html=True)
    sample_fraud = df[df['is_fraud'] == 1].head(5)
    sample_legit = df[df['is_fraud'] == 0].head(5)
    st.markdown("**🔴 Fraudulent Transactions (sample)**")
    st.dataframe(sample_fraud, use_container_width=True, hide_index=True)
    st.markdown("**🟢 Legitimate Transactions (sample)**")
    st.dataframe(sample_legit, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: ML PIPELINE                            ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "⚙️ ML Pipeline":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">⚙️ Machine Learning Pipeline</div>
        <div class="hero-subtitle">
            End-to-end ML workflow • From raw data to production-ready fraud detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Steps
    st.markdown('<div class="section-header">🔄 Pipeline Architecture</div>', unsafe_allow_html=True)

    steps = [
        ("1", "Data Generation", "200K synthetic transactions with realistic fraud patterns (2% fraud rate)"),
        ("2", "Feature Engineering", f"{len(pipeline['feature_names'])} features: temporal, behavioral, interaction, and derived"),
        ("3", "Encoding & Scaling", "Label encoding for categoricals, RobustScaler for numericals (outlier-resistant)"),
        ("4", "Stratified Split", f"80/20 train-test split preserving class distribution"),
        ("5", "SMOTE Resampling", "Oversample minority class on training set ONLY (test set untouched)"),
        ("6", "Model Training", "Logistic Regression (baseline) + XGBoost (primary) with hyperparameter tuning"),
        ("7", "Evaluation", "Precision, Recall, F1, ROC AUC, PR AUC, MCC, Cost-Benefit Analysis"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-number">{num}</div>
            <div>
                <strong style="color: #FAFAFA;">{title}</strong><br>
                <span style="color: #8890a4; font-size: 0.9rem;">{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # SMOTE Visualization
    st.markdown('<div class="section-header">⚖️ SMOTE — Before & After Class Distribution</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    pre = pipeline['pre_smote_distribution']
    post = pipeline['post_smote_distribution']

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        labels = ['Legitimate', 'Fraud']
        values = [pre.get(0, 0), pre.get(1, 0)]
        bars = ax.bar(labels, values, color=['#00D4AA', '#FF6B6B'],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                    f'{val:,}', ha='center', va='bottom', color='white', fontweight='bold')
        ax.set_title('BEFORE SMOTE (Imbalanced)', color='#FF6B6B', fontweight='bold', fontsize=13)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        values = [post.get(0, 0), post.get(1, 0)]
        bars = ax.bar(labels, values, color=['#00D4AA', '#FF6B6B'],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                    f'{val:,}', ha='center', va='bottom', color='white', fontweight='bold')
        ax.set_title('AFTER SMOTE (Balanced)', color='#00D4AA', fontweight='bold', fontsize=13)
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.1, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.info("""
    ⚠️ **Critical Design Decision**: SMOTE is applied **only to the training set**. The test set
    remains untouched to reflect real-world class distribution. This prevents data leakage and
    ensures evaluation metrics are realistic. Many projects get this wrong!
    """)

    # Feature list
    st.markdown('<div class="section-header">📐 Engineered Features</div>', unsafe_allow_html=True)

    feature_df = pd.DataFrame({
        'Feature': pipeline['feature_names'],
        'Index': range(len(pipeline['feature_names']))
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: MODEL PERFORMANCE                      ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🤖 Model Performance & Comparison</div>
        <div class="hero-subtitle">
            Logistic Regression vs XGBoost • Comprehensive evaluation on imbalanced test set
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    st.markdown('<div class="section-header">📊 Performance Comparison Table</div>', unsafe_allow_html=True)
    comparison_df = generate_comparison_table([lr_eval, xgb_eval, xgb_eval_optimal])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Model Comparison Bar Chart
    st.markdown('<div class="section-header">📊 Key Metrics Comparison</div>', unsafe_allow_html=True)
    fig = plot_model_comparison_bar([lr_eval, xgb_eval])
    st.pyplot(fig)
    plt.close(fig)

    # Confusion Matrices
    st.markdown('<div class="section-header">🔢 Confusion Matrices</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_confusion_matrix(lr_eval)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = plot_confusion_matrix(xgb_eval)
        st.pyplot(fig)
        plt.close(fig)

    # ROC & PR Curves
    st.markdown('<div class="section-header">📈 ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_roc_curves([lr_eval, xgb_eval])
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = plot_precision_recall_curves([lr_eval, xgb_eval])
        st.pyplot(fig)
        plt.close(fig)

    # Threshold Analysis
    st.markdown('<div class="section-header">🎚️ XGBoost Threshold Optimization</div>', unsafe_allow_html=True)
    st.markdown(f"**Optimal Threshold: {xgb_threshold['optimal_threshold']:.2f}** "
                f"(F1 = {xgb_threshold['best_f1']:.4f})")

    fig = plot_threshold_analysis(xgb_threshold['threshold_results'], 'XGBoost')
    st.pyplot(fig)
    plt.close(fig)

    # Why accuracy is not enough
    st.markdown('<div class="section-header">⚠️ Why Accuracy is NOT Enough</div>', unsafe_allow_html=True)
    st.warning(f"""
    **Accuracy = {xgb_eval['accuracy']:.4f}** — Looks great, right? **WRONG.**

    With a {summary['fraud_rate']:.2f}% fraud rate, a model that predicts "NOT FRAUD" for every single transaction
    would achieve **{100 - summary['fraud_rate']:.2f}% accuracy** — while catching **ZERO** fraud!

    This is why we focus on:
    - **Recall (Fraud)**: What % of actual fraud do we catch? → **{xgb_eval['recall_fraud']*100:.1f}%**
    - **Precision (Fraud)**: What % of our fraud alerts are real? → **{xgb_eval['precision_fraud']*100:.1f}%**
    - **F1-Score**: Harmonic mean of precision & recall → **{xgb_eval['f1_fraud']:.4f}**
    - **PR AUC**: Area under precision-recall curve → **{xgb_eval['pr_auc']:.4f}** (much more informative than ROC AUC for imbalanced data)
    """)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: EXPLAINABILITY                         ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🔍 Explainability & Insights":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔍 Explainability & Business Insights</div>
        <div class="hero-subtitle">
            Understanding what drives fraud • Actionable intelligence for security teams
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-header">🎯 Feature Importance — XGBoost</div>', unsafe_allow_html=True)

    xgb_imp = get_xgboost_feature_importance(xgb_result['model'], pipeline['feature_names'])
    fig = plot_feature_importance(xgb_imp, 'XGBoost Feature Importance (Gain)')
    st.pyplot(fig)
    plt.close(fig)

    # LR Feature Importance
    st.markdown('<div class="section-header">📏 Feature Importance — Logistic Regression (Coefficients)</div>', unsafe_allow_html=True)
    lr_imp = get_sklearn_feature_importance(lr_result['model'], pipeline['feature_names'])
    fig = plot_feature_importance(lr_imp, 'Logistic Regression — Absolute Coefficient Values')
    st.pyplot(fig)
    plt.close(fig)

    # Permutation Importance
    st.markdown('<div class="section-header">🔀 Permutation Importance (Model-Agnostic)</div>', unsafe_allow_html=True)
    with st.spinner("Computing permutation importance..."):
        perm_imp = compute_permutation_importance(
            xgb_result['model'], pipeline['X_test'], pipeline['y_test'],
            pipeline['feature_names'], n_repeats=5
        )
    fig = plot_permutation_importance(perm_imp, 'XGBoost Permutation Importance')
    st.pyplot(fig)
    plt.close(fig)

    # Business Insights
    st.markdown('<div class="section-header">💡 AI-Generated Business Insights</div>', unsafe_allow_html=True)

    insights = generate_business_insights(df, xgb_imp, xgb_eval)
    for insight in insights:
        severity_class = insight.get('severity', '')
        st.markdown(f"""
        <div class="insight-card {severity_class}">
            <div class="insight-title">{insight['icon']} {insight['title']}</div>
            <div class="insight-text">{insight['insight']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Importance Table
    st.markdown('<div class="section-header">📊 Full Feature Importance Table</div>', unsafe_allow_html=True)
    st.dataframe(xgb_imp, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     PAGE: LIVE PREDICTION                        ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🎯 Live Prediction":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🎯 Live Fraud Prediction Engine</div>
        <div class="hero-subtitle">
            Enter transaction details to get instant fraud risk assessment with AI explanations
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input Form
    st.markdown('<div class="section-header">📝 Transaction Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("💵 Transaction Amount ($)", min_value=0.5, max_value=50000.0,
                                 value=250.0, step=10.0, key="pred_amount")
        merchant = st.selectbox("🏪 Merchant Category", MERCHANT_CATEGORIES, key="pred_merchant")
        card_type = st.selectbox("💳 Card Type", CARD_TYPES, key="pred_card")
        entry_mode = st.selectbox("📱 Entry Mode", ENTRY_MODES, key="pred_entry")

    with col2:
        country = st.selectbox("🌍 Transaction Country", sorted(set(COUNTRIES)), key="pred_country")
        hour = st.slider("🕐 Hour of Day", 0, 23, 14, key="pred_hour")
        day = st.slider("📅 Day of Week (0=Mon, 6=Sun)", 0, 6, 2, key="pred_day")
        is_weekend = 1 if day >= 5 else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0

    with col3:
        velocity = st.slider("⚡ Transactions in 24h", 0, 15, 2, key="pred_velocity")
        distance = st.slider("📍 Distance from Home (0-1)", 0.0, 1.0, 0.1, step=0.05, key="pred_distance")
        is_cross_border = st.selectbox("🌐 Cross-Border?", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No", key="pred_cross")
        time_since_last = st.number_input("⏱️ Hours Since Last Txn", 0.0, 999.0, 12.0, key="pred_time")

    if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
        # Build transaction dict
        txn = {
            'amount': amount,
            'merchant_category': merchant,
            'card_type': card_type,
            'entry_mode': entry_mode,
            'country': country,
            'city': 'Unknown',
            'hour_of_day': hour,
            'day_of_week': day,
            'is_weekend': is_weekend,
            'is_night': is_night,
            'customer_txn_count': velocity,
            'rolling_avg_amount': amount * 0.6,
            'amount_deviation': 0.5,
            'time_since_last_txn': time_since_last,
            'velocity_24h': velocity,
            'is_cross_border': is_cross_border,
            'distance_from_home': distance,
        }

        with st.spinner("🤖 AI analyzing transaction..."):
            time.sleep(0.5)  # Dramatic effect

            txn_df = preprocess_single_transaction(
                txn, pipeline['scaler'], pipeline['encoders'], pipeline['feature_names']
            )

            # Get predictions from both models
            lr_proba = lr_result['model'].predict_proba(txn_df)[0][1]
            xgb_proba = xgb_result['model'].predict_proba(txn_df)[0][1]
            xgb_pred = 1 if xgb_proba >= xgb_threshold['optimal_threshold'] else 0

        st.markdown("---")

        # Results
        st.markdown('<div class="section-header">🎯 Analysis Results</div>', unsafe_allow_html=True)

        if xgb_pred == 1:
            st.markdown(f"""
            <div class="kpi-card" style="border: 2px solid #FF6B6B;">
                <div class="kpi-label">VERDICT</div>
                <div class="kpi-value kpi-red">🚨 FRAUD DETECTED</div>
                <div class="kpi-delta" style="color: #FF6B6B;">
                    This transaction has been flagged as potentially fraudulent
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-card" style="border: 2px solid #00D4AA;">
                <div class="kpi-label">VERDICT</div>
                <div class="kpi-value kpi-green">✅ LEGITIMATE</div>
                <div class="kpi-delta" style="color: #00D4AA;">
                    This transaction appears to be legitimate
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Probability Scores
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">XGBoost Fraud Probability</div>
                <div class="kpi-value {'kpi-red' if xgb_proba > 0.5 else 'kpi-green'}">{xgb_proba*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">LR Fraud Probability</div>
                <div class="kpi-value {'kpi-red' if lr_proba > 0.5 else 'kpi-green'}">{lr_proba*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            risk_level = "🔴 HIGH" if xgb_proba > 0.7 else ("🟡 MEDIUM" if xgb_proba > 0.3 else "🟢 LOW")
            risk_color = "kpi-red" if xgb_proba > 0.7 else ("kpi-yellow" if xgb_proba > 0.3 else "kpi-green")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value {risk_color}">{risk_level}</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factors
        st.markdown('<div class="section-header">🔎 Risk Factor Breakdown</div>', unsafe_allow_html=True)

        factors = explain_single_prediction(
            xgb_result['model'], txn_df, pipeline['feature_names'], xgb_pred, xgb_proba
        )

        if factors:
            factor_df = pd.DataFrame(factors)
            st.dataframe(factor_df, use_container_width=True, hide_index=True)

        # Risk visualization
        st.markdown("")
        risk_pct = min(int(xgb_proba * 100), 100)
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #00D4AA; font-size: 0.85rem;">Low Risk</span>
                <span style="color: #FFE66D; font-size: 0.85rem;">Medium</span>
                <span style="color: #FF6B6B; font-size: 0.85rem;">High Risk</span>
            </div>
            <div class="risk-meter">
                <div class="risk-indicator" style="left: {risk_pct}%;"></div>
            </div>
            <div style="text-align: center; color: #8890a4; font-size: 0.85rem; margin-top: 0.5rem;">
                Fraud Probability: {xgb_proba*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
