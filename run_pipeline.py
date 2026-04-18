"""
Run the complete ML pipeline end-to-end.
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.data_generator import generate_dataset, get_dataset_summary
from src.preprocessing import preprocess_pipeline
from src.model import train_logistic_regression, train_xgboost, find_optimal_threshold, save_models
from src.evaluation import evaluate_model, generate_comparison_table

print('=' * 60)
print('LOADING DATA...')
df = pd.read_csv('data/transactions.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'Data loaded: {df.shape}')

print('=' * 60)
print('PREPROCESSING...')
pipeline = preprocess_pipeline(df)

print('=' * 60)
print('TRAINING MODELS...')
lr_result = train_logistic_regression(pipeline['X_train_resampled'], pipeline['y_train_resampled'])
xgb_result = train_xgboost(pipeline['X_train_resampled'], pipeline['y_train_resampled'],
                           pipeline['X_test'], pipeline['y_test'])

print('=' * 60)
print('EVALUATING...')
lr_eval = evaluate_model(lr_result['model'], pipeline['X_test'], pipeline['y_test'], 'Logistic Regression')
xgb_eval = evaluate_model(xgb_result['model'], pipeline['X_test'], pipeline['y_test'], 'XGBoost')

print('=' * 60)
print('THRESHOLD OPTIMIZATION...')
threshold = find_optimal_threshold(xgb_result['model'], pipeline['X_test'], pipeline['y_test'])
print('Optimal threshold:', threshold['optimal_threshold'])
print('Best F1:', threshold['best_f1'])

print('=' * 60)
print('SAVING MODELS...')
save_models(lr_result, xgb_result, pipeline['scaler'], pipeline['encoders'], pipeline['feature_names'])

print('=' * 60)
print('COMPARISON TABLE:')
comp = generate_comparison_table([lr_eval, xgb_eval])
print(comp.to_string())
print('=' * 60)
print('ALL PIPELINE STEPS COMPLETE!')
