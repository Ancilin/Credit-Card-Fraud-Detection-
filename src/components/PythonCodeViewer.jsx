import React, { useState } from 'react';
import { Code2, Copy, Check, FileText, Download, Terminal } from 'lucide-react';

export default function PythonCodeViewer() {
  const [activeFile, setActiveFile] = useState('train_models.py');
  const [copied, setCopied] = useState(false);

  const files = {
    'train_models.py': `
"""
Credit Card Fraud Detection - Model Training & SMOTE Resampling Pipeline
Author: Antigravity Machine Learning Showcase
Dataset: Kaggle Credit Card Fraud Detection (284,807 transactions, 0.172% fraud)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib

def load_and_preprocess_data(filepath="data/creditcard.csv"):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Robust scaling for Amount and Time (outlier resistance)
    scaler = RobustScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    return X, y, scaler

def train_xgb_smote_pipeline(X_train, y_train):
    """
    Train XGBoost with SMOTE within Imblearn Pipeline to eliminate Data Leakage
    """
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    pipeline = ImbPipeline([
        ('smote', SMOTE(sampling_strategy=0.2, random_state=42)),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

if __name__ == "__main__":
    from generate_data import generate_synthetic_kaggle_data
    
    print("Generating synthetic Kaggle dataset...")
    df = generate_synthetic_kaggle_data(sample_count=20000, fraud_ratio=0.0017)
    
    X = df.drop(columns=['Class', 'id'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Train shapes: {X_train.shape}, Test shapes: {X_test.shape}")
    print(f"Train fraud count: {y_train.sum()} ({y_train.mean()*100:.3f}%)")

    model = train_xgb_smote_pipeline(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\\n--- Model Evaluation ---")
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
    print("PR-AUC Score (Average Precision):", average_precision_score(y_test, y_pred_proba))
`,
    'generate_data.py': `
"""
Kaggle Schema Credit Card Fraud Dataset Synthetic Generator
Generates realistic 28 PCA features (V1-V28), Time, Amount, and Class
"""

import numpy as np
import pandas as pd

def generate_synthetic_kaggle_data(sample_count=50000, fraud_ratio=0.00172, random_state=42):
    np.random.seed(random_state)
    fraud_count = int(sample_count * fraud_ratio)
    normal_count = sample_count - fraud_count

    # Generate normal transactions
    normal_data = {
        'Time': np.random.uniform(0, 172800, normal_count),
        'Amount': np.random.exponential(scale=88.3, size=normal_count),
        'Class': np.zeros(normal_count, dtype=int)
    }
    for i in range(1, 29):
        normal_data[f'V{i}'] = np.random.normal(0.0, 1.0, normal_count)

    # Inject fraud patterns (strong mean shifts in V14, V12, V10, V4, V11)
    fraud_data = {
        'Time': np.random.uniform(0, 172800, fraud_count),
        'Amount': np.random.choice([np.random.normal(1.25, 0.5), np.random.normal(850.0, 300)], size=fraud_count),
        'Class': np.ones(fraud_count, dtype=int)
    }
    for i in range(1, 29):
        fraud_data[f'V{i}'] = np.random.normal(0.0, 1.0, fraud_count)
    
    # Specific PCA anomaly signatures
    fraud_data['V14'] = np.random.normal(-4.5, 1.2, fraud_count)
    fraud_data['V12'] = np.random.normal(-3.8, 1.1, fraud_count)
    fraud_data['V10'] = np.random.normal(-3.2, 1.0, fraud_count)
    fraud_data['V4']  = np.random.normal(3.6, 1.1, fraud_count)
    fraud_data['V11'] = np.random.normal(2.9, 1.0, fraud_count)

    df_normal = pd.DataFrame(normal_data)
    df_fraud = pd.DataFrame(fraud_data)

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df['id'] = [f'TX-{i:06d}' for i in range(len(df))]

    return df

if __name__ == "__main__":
    df = generate_synthetic_kaggle_data()
    print("Generated Dataset Shape:", df.shape)
    print("Class Distribution:\\n", df['Class'].value_counts(normalize=True))
    df.to_csv("creditcard_synthetic.csv", index=False)
`,
    'evaluate.py': `
"""
Financial Cost Matrix & Threshold Optimization Module
"""

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

def evaluate_cost_matrix(y_true, y_pred_proba, cost_fn=500, cost_fp=15):
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

        results.append({
            'threshold': t,
            'total_cost': total_cost,
            'precision': precision,
            'recall': recall,
            'f2_score': f2,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })

    best_result = min(results, key=lambda x: x['total_cost'])
    return best_result, results
`,
    'requirements.txt': `
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
lightgbm>=3.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
`
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(files[activeFile] || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      
      {/* Title Header */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Code2 className="w-5 h-5 text-indigo-400" />
            <span>Python ML Pipeline & Jupyter Code Base</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Modular Python scripts for dataset generation, SMOTE pipelines, cost matrices, and evaluation.
          </p>
        </div>

        {/* Copy Script Button */}
        <button
          onClick={handleCopy}
          className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-all cursor-pointer"
        >
          {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4 text-indigo-400" />}
          <span>{copied ? 'Copied to Clipboard!' : 'Copy File Content'}</span>
        </button>
      </div>

      {/* Code File Viewer */}
      <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
        
        {/* File Tabs */}
        <div className="flex overflow-x-auto bg-slate-950/80 border-b border-slate-800 px-3 py-2 space-x-2 no-scrollbar">
          {Object.keys(files).map((filename) => (
            <button
              key={filename}
              onClick={() => setActiveFile(filename)}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                activeFile === filename
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              <span>{filename}</span>
            </button>
          ))}
        </div>

        {/* Code View Body */}
        <div className="p-4 bg-slate-950 text-slate-200 font-mono text-xs overflow-x-auto max-h-[500px]">
          <pre className="leading-relaxed">
            <code>{files[activeFile]}</code>
          </pre>
        </div>

      </div>

    </div>
  );
}
