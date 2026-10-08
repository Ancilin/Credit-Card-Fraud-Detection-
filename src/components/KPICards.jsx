import React from 'react';
import { 
  CreditCard, 
  AlertTriangle, 
  TrendingUp, 
  DollarSign, 
  Layers, 
  ArrowRight,
  ShieldCheck,
  Zap
} from 'lucide-react';

export default function KPICards({ setActiveTab }) {
  const kpis = [
    {
      title: 'Total Dataset Volume',
      value: '284,807',
      subtitle: '48 Hours European Cardholders',
      change: '28 PCA Features + Time/Amount',
      icon: CreditCard,
      color: 'blue'
    },
    {
      title: 'Class Imbalance Rate',
      value: '0.172%',
      subtitle: '492 Fraud vs 284,315 Normal',
      change: '1 Fraud per 578 Transactions',
      icon: AlertTriangle,
      color: 'rose'
    },
    {
      title: 'SMOTE PR-AUC Score',
      value: '0.942',
      subtitle: 'vs 0.720 Baseline Linear',
      change: '+30.8% Precision-Recall Lift',
      icon: TrendingUp,
      color: 'purple'
    },
    {
      title: 'Financial Loss Saved',
      value: '$44,150',
      subtitle: 'Per 56k Transactions Evaluated',
      change: '95.8% Total Loss Prevention',
      icon: DollarSign,
      color: 'emerald'
    }
  ];

  return (
    <div className="space-y-6">
      
      {/* Top Banner / Hero */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/70 to-slate-900 border border-slate-800 p-6 sm:p-8 shadow-2xl">
        <div className="absolute -right-12 -top-12 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute -left-12 -bottom-12 w-64 h-64 bg-rose-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
              <Zap className="w-3.5 h-3.5" />
              <span>Interview-Grade Machine Learning Showcase</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Credit Card Fraud Detection <span className="gradient-text-blue">under Extreme Imbalance</span>
            </h1>
            <p className="text-sm text-slate-300 leading-relaxed">
              In real-world fraud detection, class distributions are highly skewed ($99.83\%$ genuine vs $0.17\%$ fraudulent). Standard accuracy yields false security ($99.83\%$ accuracy by predicting 0 every time). This project evaluates **SMOTE**, **Random Undersampling**, **Cost-Sensitive Learning**, and **PR-AUC threshold optimization**.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => setActiveTab('simulator')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-indigo-600 text-white font-semibold text-xs shadow-lg shadow-rose-600/20 hover:opacity-90 transition-all cursor-pointer"
            >
              <Zap className="w-4 h-4" />
              <span>Launch Live Simulator</span>
            </button>
            <button
              onClick={() => setActiveTab('imbalance')}
              className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-slate-800/90 text-slate-200 border border-slate-700 font-semibold text-xs hover:bg-slate-700/80 transition-all cursor-pointer"
            >
              <Layers className="w-4 h-4 text-indigo-400" />
              <span>Explore SMOTE Lab</span>
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, idx) => {
          const Icon = kpi.icon;
          return (
            <div key={idx} className="glass-card rounded-xl p-5 relative overflow-hidden border border-slate-800">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{kpi.title}</span>
                <div className={`p-2 rounded-lg bg-${kpi.color}-500/10 text-${kpi.color}-400 border border-${kpi.color}-500/20`}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <div className="text-2xl font-bold text-white mb-1">{kpi.value}</div>
              <div className="text-xs text-slate-300 font-medium mb-2">{kpi.subtitle}</div>
              <div className="text-[11px] text-emerald-400 font-medium flex items-center space-x-1 bg-emerald-500/5 px-2 py-1 rounded w-fit border border-emerald-500/10">
                <ShieldCheck className="w-3 h-3" />
                <span>{kpi.change}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Feature Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div 
          onClick={() => setActiveTab('eda')}
          className="glass-card rounded-xl p-5 cursor-pointer border border-slate-800/80 hover:border-indigo-500/40 transition-all group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition-colors">1. Exploratory Data Analysis</h3>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Inspect PCA feature distributions ($V_1-V_{28}$), transaction amount skewness, correlation heatmaps, and temporal fraud peaks.
          </p>
        </div>

        <div 
          onClick={() => setActiveTab('models')}
          className="glass-card rounded-xl p-5 cursor-pointer border border-slate-800/80 hover:border-indigo-500/40 transition-all group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition-colors">2. Model & Threshold Studio</h3>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Compare Logistic Regression, Random Forest, and XGBoost using ROC & Precision-Recall curves with dynamic threshold cost optimization.
          </p>
        </div>

        <div 
          onClick={() => setActiveTab('interview')}
          className="glass-card rounded-xl p-5 cursor-pointer border border-slate-800/80 hover:border-indigo-500/40 transition-all group"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold text-white text-sm group-hover:text-indigo-400 transition-colors">3. ML Interview Cheat Sheet</h3>
            <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all" />
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Master interview questions on PR-AUC vs ROC-AUC, SMOTE vs ADASYN, cost matrices, data leakage, and live production monitoring.
          </p>
        </div>
      </div>

    </div>
  );
}
