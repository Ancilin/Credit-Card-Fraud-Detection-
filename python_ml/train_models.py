"""
Credit Card Fraud Detection - Model Training & Imbalance Benchmark Script
Author: Antigravity Machine Learning Showcase
"""

import numpy as np
import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb

# Import local data generator and evaluator
from generate_data import generate_synthetic_kaggle_data
from evaluate import evaluate_classifier, optimize_threshold_cost

def main():
    print("=" * 70)
    print(" CREDIT CARD FRAUD DETECTION - IMBALANCED CLASSIFICATION BENCHMARK ")
    print("=" * 70)

    # 1. Load / Generate Dataset
    csv_path = "data/creditcard_synthetic.csv"
    if not os.path.exists(csv_path):
        print("Generating synthetic Kaggle-schema dataset...")
        df = generate_synthetic_kaggle_data(sample_count=20000, fraud_ratio=0.002)
    else:
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

    # 2. Preprocessing & Scaler
    drop_cols = ['Class']
    if 'id' in df.columns:
        drop_cols.append('id')

    X = df.drop(columns=drop_cols)
    y = df['Class']

    scaler = RobustScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    # 3. Stratified Train-Test Split (Crucial for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"\nDataset Splits:")
    print(f"  X_train: {X_train.shape}, Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"  X_test:  {X_test.shape}, Fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    # 4. Model Configurations & Pipelines
    models = {
        'Baseline Logistic Regression (No Resampling)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression + SMOTE': ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest + SMOTE': ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
        ]),
        'XGBoost + SMOTE (Scale Pos Weight)': ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
            ('clf', xgb.XGBClassifier(
                n_estimators=150, 
                max_depth=5, 
                learning_rate=0.05,
                scale_pos_weight=(len(y_train[y_train==0])/len(y_train[y_train==1])),
                eval_metric='aucpr',
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # 5. Train & Evaluate
    results = []
    print("\nTraining models and calculating metrics...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

        metrics = evaluate_classifier(y_test, y_proba, threshold=0.5)
        cost_opt, _ = optimize_threshold_cost(y_test, y_proba)

        results.append({
            'Model': name,
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'F2-Score': f"{metrics['f2_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'PR-AUC (Key Metric)': f"{metrics['pr_auc']:.4f}",
            'Opt Threshold': f"{cost_opt['threshold']:.2f}",
            'Financial Savings ($)': f"${cost_opt['savings']:,.2f}"
        })

    # Display Results Table
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print(" BENCHMARK RESULTS TABLE ")
    print("=" * 90)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
