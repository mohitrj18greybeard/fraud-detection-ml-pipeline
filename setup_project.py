"""
FraudShield AI - Setup & Training Script.
Generates initial data and trains the production model.
"""

import os
import joblib
import pandas as pd
from src.data_generator import generate_dataset
from src.preprocessing import preprocess_pipeline
from src.model import train_xgboost, save_models

def main():
    print("Initializing FraudShield AI Project...")
    
    # 1. Create directories
    for folder in ["data", "models"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}")
            
    # 2. Generate Data
    data_path = "data/transactions.csv"
    if not os.path.exists(data_path):
        print("Generating synthetic dataset (200k samples)...")
        df = generate_dataset(n_samples=200000, save_path=data_path)
    else:
        print("Dataset already exists.")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # 3. Preprocess
    print("Preprocessing data and applying SMOTE...")
    pipeline = preprocess_pipeline(df)
    
    # 4. Train
    print("Training XGBoost Production Model...")
    xgb_result = train_xgboost(
        pipeline['X_train_resampled'], 
        pipeline['y_train_resampled'],
        pipeline['X_test'],
        pipeline['y_test']
    )
    
    # 5. Save
    print("Saving model to models/model.pkl...")
    joblib.dump(xgb_result['model'], "models/model.pkl")
    
    print("Project setup complete! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()
