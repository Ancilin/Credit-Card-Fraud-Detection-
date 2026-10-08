"""
Financial Cost Matrix & Imbalanced Evaluation Metrics Module
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    fbeta_score, 
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve
)

def evaluate_classifier(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    return {
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score': f2,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }

def optimize_threshold_cost(y_true, y_pred_proba, cost_fn=500, cost_fp=15):
    """
    Sweeps probability thresholds to find threshold minimizing Total Financial Cost
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0

        results.append({
            'threshold': t,
            'total_cost': total_cost,
            'precision': p,
            'recall': r,
            'f2_score': f2,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        })

    best_result = min(results, key=lambda x: x['total_cost'])
    baseline_cost = np.sum(y_true == 1) * cost_fn
    
    best_result['baseline_cost'] = baseline_cost
    best_result['savings'] = baseline_cost - best_result['total_cost']
    best_result['savings_pct'] = (best_result['savings'] / baseline_cost) * 100
    
    return best_result, pd.DataFrame(results)

if __name__ == "__main__":
    y_true = np.array([0]*990 + [1]*10)
    y_proba = np.random.uniform(0, 1, 1000)
    best, df_res = optimize_threshold_cost(y_true, y_proba)
    print("Optimal Threshold:", best['threshold'])
    print("Min Financial Loss:", best['total_cost'])
