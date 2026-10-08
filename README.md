# 💳 Credit Card Fraud Detection (Imbalanced Classification ML Showcase)

[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python)](https://python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%2B%20XGBoost-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, interview-grade **Credit Card Fraud Detection System** featuring interactive web-based data visualization, real-time transaction simulation, ML imbalanced classification pipelines (**SMOTE**, **Cost-Weighted XGBoost**, **Random Forest**), financial cost matrix optimization, and a senior ML interview study guide.

---

## 🚀 Key Features & Highlights

- **📊 Exploratory Data Analysis (EDA) Hub**: Interactive visual density distributions for 28 PCA features ($V_1-V_{28}$), transaction amount skewness, feature-target correlations, and 48-hour temporal fraud trends.
- **🔄 Class Imbalance & Resampling Lab**: Side-by-side comparative visualizer for **Baseline Unbalanced (0.17%)** vs **SMOTE Oversampling (50:50)** vs **Random Undersampling** vs **Cost-Sensitive Class Weighting**, complete with a 2D PCA feature scatter plot.
- **🎛️ Model Studio & Decision Threshold Optimizer**: 
  - Compare Logistic Regression, Random Forest, XGBoost, Neural Networks, and Isolation Forest.
  - Multi-model ROC and Precision-Recall (PR-AUC) curves.
  - Dynamic decision threshold slider ($t \in [0.05, 0.95]$) calculating Confusion Matrix ($TP, FN, FP, TN$) and Financial Loss Saved ($Cost = C_{FN} \times FN + C_{FP} \times FP$).
- **⚡ Real-Time Fraud Simulator & SHAP Visualizer**: 
  - Live probability risk engine with preset transaction scenarios (Grocery Store, High-Amount Drain, Bot Micro-Charge, Luxury Purchase).
  - SHAP-like feature attribution waterfall chart explaining positive and negative risk factors.
- **🎓 ML Interview Mastery Guide**: Deep-dive Q&A covering Accuracy traps, ROC-AUC vs PR-AUC, SMOTE algorithms, data leakage prevention, and cost matrices.
- **🐍 Modular Python ML Pipeline & Jupyter Notebook**: Complete scikit-learn & XGBoost training scripts (`python_ml/train_models.py`) and portfolio notebook (`python_ml/notebooks/credit_card_fraud_detection.ipynb`).

---

## 🛠️ Tech Stack

### Web Application
- **Framework**: React 19 + Vite 6
- **Styling**: Tailwind CSS v4 + Vanilla CSS Glassmorphism
- **Charts & Motion**: Recharts + Canvas Confetti
- **Icons**: Lucide React

### Machine Learning Engine
- **Languages & Libraries**: Python 3.14, Scikit-Learn, Imbalanced-Learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
- **Data Schema**: Kaggle Credit Card Fraud Detection (284,807 transactions, 28 PCA features + Time & Amount)

---

## 📂 Repository Structure

```text
├── src/
│   ├── components/
│   │   ├── Navbar.jsx          # Header navigation bar
│   │   ├── KPICards.jsx        # Overview stat metrics & hero banner
│   │   ├── EDADashboard.jsx    # Interactive PCA feature distribution & temporal charts
│   │   ├── ImbalanceLab.jsx    # Resampling playground (SMOTE vs Undersampling vs Class Weight)
│   │   ├── ModelStudio.jsx     # ROC/PR curves & threshold cost optimizer
│   │   ├── LiveSimulator.jsx   # Live transaction tester & SHAP waterfall chart
│   │   ├── InterviewGuide.jsx  # ML interview Q&A cards & math breakdowns
│   │   └── PythonCodeViewer.py # Embedded Python script inspector
│   ├── ml/
│   │   └── fraudEngine.js      # Client-side synthetic dataset generator & ML inference
│   ├── App.jsx                 # Main React container
│   └── index.css               # Tailwind & glassmorphism custom styles
├── python_ml/
│   ├── generate_data.py        # Synthetic Kaggle-schema generator
│   ├── evaluate.py             # Precision, Recall, PR-AUC & Financial Cost Matrix
│   ├── train_models.py         # Full ML training pipeline (Baseline, SMOTE, XGBoost)
│   ├── requirements.txt        # Python dependencies
│   └── notebooks/
│       └── credit_card_fraud_detection.ipynb # Portfolio Jupyter Notebook
├── package.json
└── README.md
```

---

## 💻 Quick Start & Installation

### 1. Web Application (React + Vite)

```bash
# Clone the repository
git clone https://github.com/Ancilin/Credit-Card-Fraud-Detection-.git
cd Credit-Card-Fraud-Detection-

# Install npm dependencies
npm install

# Start local dev server
npm run dev
```
Open your browser at `http://localhost:5173`.

---

### 2. Python ML Pipeline & Notebook

```bash
# Navigate to project directory
cd python_ml

# Install Python requirements
pip install -r requirements.txt

# Run Python training benchmark
python train_models.py
```

#### (Optional) Training on Real Kaggle `creditcard.csv`
1. Download `creditcard.csv` from [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place `creditcard.csv` inside `python_ml/data/creditcard.csv`.
3. Re-run `python train_models.py` (the script automatically detects and uses the real Kaggle CSV!).

---

## 📈 ML Benchmark Performance Summary

| Model Configuration | Precision | Recall | $F_1$-Score | $F_2$-Score | ROC-AUC | PR-AUC (Key Metric) | Financial Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $\$5,000.00$ |
| **Logistic Regression + SMOTE** | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $1.0000$ | $\$5,000.00$ |
| **Random Forest + SMOTE** | $1.0000$ | $0.9000$ | $0.9474$ | $0.9184$ | $1.0000$ | $1.0000$ | $\$5,000.00$ |
| **XGBoost + SMOTE (Scale Pos Weight)** | $1.0000$ | $0.9000$ | $0.9474$ | $0.9184$ | $0.9998$ | $0.9455$ | $\$4,500.00$ |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

⭐ **If you found this project helpful, please give it a Star on GitHub!**
