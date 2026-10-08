import React, { useState } from 'react';
import { 
  GraduationCap, 
  ChevronDown, 
  ChevronUp, 
  Code2
} from 'lucide-react';

export default function InterviewGuide() {
  const [openIdx, setOpenIdx] = useState(0);

  const questions = [
    {
      title: "1. Why is Accuracy a misleading metric for Credit Card Fraud Detection?",
      summary: "In a dataset with 0.17% fraud, a dummy model predicting 0 every time achieves 99.83% accuracy while catching 0% of fraud.",
      detail: `In severely imbalanced classification tasks (99.83% majority vs 0.17% minority), standard classification accuracy:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

is dominated by True Negatives (TN). A naive baseline model that blindly predicts Class 0 for every transaction will achieve 99.83% accuracy. However, its Recall (Sensitivity) is 0%, allowing 100% of fraudulent transactions to pass through undetected.

Always evaluate imbalanced classifiers using Precision-Recall AUC (PR-AUC / Average Precision), F-beta score (e.g., F2 score to penalize False Negatives), and Confusion Matrices.`,
      code: `# Python Example: Misleading Accuracy Baseline vs Recall
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score
import numpy as np

y_true = np.array([0]*9983 + [1]*17)  # 0.17% fraud
y_pred_naive = np.zeros_like(y_true)   # Predict all 0s

print("Naive Accuracy:", accuracy_score(y_true, y_pred_naive))      # Output: 0.9983 (99.83%)
print("Naive Recall:", recall_score(y_true, y_pred_naive))          # Output: 0.0000 (0%)
print("PR-AUC Score:", average_precision_score(y_true, y_pred_naive))# Output: 0.0017`
    },
    {
      title: "2. What is the difference between ROC-AUC and Precision-Recall AUC (PR-AUC)?",
      summary: "ROC-AUC uses False Positive Rate (FP/TN), which remains tiny under large TN volume, whereas PR-AUC compares Precision directly against Recall.",
      detail: `ROC Curve plots True Positive Rate (TPR = TP / (TP + FN)) vs False Positive Rate (FPR = FP / (FP + TN)). Because TN is massive (284,315), even if FP = 1,000, FPR is ~0.0035, making ROC look artificially optimistic (ROC-AUC > 0.98).

PR Curve plots Precision (P = TP / (TP + FP)) vs Recall (R = TP / (TP + FN)). Precision directly evaluates how clean your fraud alerts are without being diluted by huge TN counts.

Use ROC-AUC for balanced datasets or general ranking. Use PR-AUC (Average Precision) whenever positive minority cases are extremely rare (< 1%).`,
      code: `# Python Example: Plotting PR-AUC & ROC-AUC with Scikit-Learn
from sklearn.metrics import roc_auc_score, average_precision_score

roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"ROC-AUC: {roc_auc:.4f} (Often misleadingly high)")
print(f"PR-AUC:  {pr_auc:.4f} (Authoritative imbalanced metric)")`
    },
    {
      title: "3. How does SMOTE work mathematically, and how to prevent Data Leakage?",
      summary: "SMOTE generates synthetic points along line segments connecting k-nearest neighbors in feature space. SMOTE MUST ONLY be applied to training folds after splitting.",
      detail: `For each minority sample x_i in S_minority:
1. Find its k-nearest neighbors in feature space (typically k=5).
2. Randomly pick one neighbor x_zi.
3. Generate synthetic sample: x_new = x_i + λ (x_zi - x_i) where λ ~ U(0,1).

CRITICAL DATA LEAKAGE RULE:
Never apply SMOTE before Train-Test Splitting or Cross-Validation!
If SMOTE is run on the entire dataset first, synthetic samples of test points will bleed into the training set, causing catastrophic target leakage and overinflated validation performance.`,
      code: `# CORRECT Pipeline: SMOTE applied ONLY inside training split / Imblearn Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Correct CV pipeline ensuring SMOTE is executed strictly on train folds
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=6))
])

pipeline.fit(X_train, y_train)`
    },
    {
      title: "4. What is Cost-Sensitive Learning and scale_pos_weight?",
      summary: "Instead of creating synthetic data, cost-sensitive learning weights the loss function so that missing a minority sample incurs a heavier gradient penalty.",
      detail: `In standard Binary Cross-Entropy (Log-Loss), the loss function is dominated by majority negative samples.

With Cost-Sensitive Learning / scale_pos_weight:
w_1 = N_negative / N_positive ≈ 284,315 / 492 ≈ 577.8

The loss function applies weight w_1 to positive minority losses, forcing the gradient descent optimizer to update model parameters 578x harder whenever a fraud sample is misclassified.`,
      code: `# XGBoost / LightGBM Class Weighting Example
import xgboost as xgb

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # ~578x loss penalty
    max_depth=6,
    learning_rate=0.05,
    eval_metric='aucpr'
)
model.fit(X_train, y_train)`
    },
    {
      title: "5. How to optimize Decision Thresholds using a Financial Cost Matrix?",
      summary: "The default 0.5 probability threshold is rarely optimal in fraud detection. Tune threshold t to minimize Total Cost = (FN * C_FN) + (FP * C_FP).",
      detail: `Financial Cost Matrix formulation:
Let C_FN be the average financial loss of an uncaught fraud (e.g. $500), and C_FP be the administrative cost of a false alert / SMS check (e.g. $15).

Total Financial Cost(t) = FN(t) * 500 + FP(t) * 15

By sweeping threshold t ∈ [0.01, 0.99], we compute the point of minimum financial loss, which typically occurs around t ≈ 0.30 - 0.40 rather than the default 0.50.`,
      code: `# Threshold Optimization Script
import numpy as np

probabilities = model.predict_proba(X_test)[:, 1]
costs = []
thresholds = np.linspace(0.01, 0.99, 99)

for t in thresholds:
    preds = (probabilities >= t).astype(int)
    fn = np.sum((y_test == 1) & (preds == 0))
    fp = np.sum((y_test == 0) & (preds == 1))
    total_cost = (fn * 500) + (fp * 15)
    costs.append(total_cost)

best_threshold = thresholds[np.argmin(costs)]
print(f"Optimal Financial Decision Threshold: {best_threshold:.2f}")`
    }
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-semibold mb-2">
            <GraduationCap className="w-3.5 h-3.5" />
            <span>Senior ML Engineer & Interview Q&A Hub</span>
          </div>
          <h2 className="text-xl font-bold text-white">Machine Learning Interview Mastery Guide</h2>
          <p className="text-xs text-slate-400 mt-1">
            Top 5 core machine learning interview topics for imbalanced classification, metric selection, data leakage prevention, and cost optimization.
          </p>
        </div>
      </div>

      {/* Accordion Questions */}
      <div className="space-y-4">
        {questions.map((q, idx) => {
          const isOpen = openIdx === idx;
          return (
            <div 
              key={idx}
              className={`glass-panel rounded-2xl border transition-all overflow-hidden ${
                isOpen ? 'border-indigo-500/40 bg-slate-900/90' : 'border-slate-800 hover:border-slate-700 bg-slate-950/60'
              }`}
            >
              {/* Question Header */}
              <button
                onClick={() => setOpenIdx(isOpen ? null : idx)}
                className="w-full text-left p-5 flex items-start justify-between gap-4 cursor-pointer"
              >
                <div className="space-y-1">
                  <h3 className="font-bold text-white text-sm sm:text-base flex items-center space-x-2">
                    <span>{q.title}</span>
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{q.summary}</p>
                </div>

                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 shrink-0">
                  {isOpen ? <ChevronDown className="w-4 h-4 text-indigo-400 rotate-180 transition-transform" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {/* Accordion Body */}
              {isOpen && (
                <div className="p-5 pt-0 border-t border-slate-800/80 space-y-4 text-xs text-slate-300">
                  
                  <div className="space-y-3 leading-relaxed bg-slate-900/50 p-4 rounded-xl border border-slate-800/80">
                    {q.detail.split('\n\n').map((paragraph, pIdx) => (
                      <p key={pIdx} className="text-slate-300 leading-relaxed font-sans">
                        {paragraph}
                      </p>
                    ))}
                  </div>

                  {/* Code snippet block */}
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-400">
                      <Code2 className="w-4 h-4" />
                      <span>Production Python Implementation:</span>
                    </div>
                    <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-emerald-400 font-mono text-[11px] overflow-x-auto leading-relaxed">
                      <code>{q.code.trim()}</code>
                    </pre>
                  </div>

                </div>
              )}

            </div>
          );
        })}
      </div>

    </div>
  );
}
