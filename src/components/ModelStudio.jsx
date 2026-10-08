import React, { useState, useMemo } from 'react';
import { 
  getModelPerformanceSpecs, 
  generateROCCurve, 
  generatePRCurve 
} from '../ml/fraudEngine';
import { Cpu, Sliders, DollarSign, TrendingUp, ShieldAlert, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function ModelStudio() {
  const [selectedModel, setSelectedModel] = useState('xgboost');
  const [selectedResampling, setSelectedResampling] = useState('smote');
  const [threshold, setThreshold] = useState(0.50);

  // Compute live specs for selected configuration
  const spec = useMemo(
    () => getModelPerformanceSpecs(selectedModel, selectedResampling, threshold),
    [selectedModel, selectedResampling, threshold]
  );

  // Generate ROC & PR Curves for selected model
  const rocPoints = useMemo(() => generateROCCurve(spec.rocAuc), [spec.rocAuc]);
  const prPoints = useMemo(() => generatePRCurve(spec.prAuc), [spec.prAuc]);

  const models = [
    { id: 'xgboost', name: 'XGBoost Classifier', tag: 'State of Art', desc: 'Gradient boosted trees with scale_pos_weight' },
    { id: 'random_forest', name: 'Random Forest', tag: 'Ensemble Bagging', desc: 'Parallel decision trees with balanced subsample' },
    { id: 'logistic_regression', name: 'Logistic Regression', tag: 'Linear Baseline', desc: 'Interpretable baseline with L2 regularization' },
    { id: 'neural_network', name: 'MLP Neural Network', tag: 'Deep Learning', desc: 'Multi-layer perceptron with dropout' },
    { id: 'isolation_forest', name: 'Isolation Forest', tag: 'Unsupervised', desc: 'Anomaly detection isolating rare outlier paths' }
  ];

  const resamplings = [
    { id: 'smote', label: 'SMOTE Oversampling' },
    { id: 'class_weight', label: 'Cost Class Weighting' },
    { id: 'undersample', label: 'Random Undersampling' },
    { id: 'raw', label: 'Baseline Raw (Unbalanced)' }
  ];

  return (
    <div className="space-y-6">
      
      {/* Title Header */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            <span>Model Training & Threshold Optimization Studio</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Tune probability decision thresholds ($t$) and optimize financial trade-offs between False Negatives and False Positives.
          </p>
        </div>

        {/* Model Selection Tabs */}
        <div className="flex flex-wrap gap-2">
          {models.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                selectedModel === m.id
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 border border-indigo-400/40'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {m.name}
            </button>
          ))}
        </div>
      </div>

      {/* Control Strip & Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Interactive Threshold Slider & Financial Cost */}
        <div className="glass-panel p-5 rounded-2xl space-y-6">
          <div className="border-b border-slate-800/80 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-rose-400" />
              <span>Decision Threshold Slider: {threshold.toFixed(2)}</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Move slider to see real-time Precision vs Recall & Cost matrix adjustments.
            </p>
          </div>

          {/* Threshold Slider Input */}
          <div className="space-y-3 bg-slate-900/70 p-4 rounded-xl border border-slate-800">
            <div className="flex justify-between text-xs font-semibold text-slate-300">
              <span>0.05 (High Recall / Catch All)</span>
              <span className="text-indigo-400 font-bold text-sm">t = {threshold.toFixed(2)}</span>
              <span>0.95 (High Precision)</span>
            </div>
            
            <input 
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg"
            />

            <div className="flex justify-between text-[11px] text-slate-400 pt-1">
              <span>Lower Threshold $\rightarrow$ Fewer Missed Frauds</span>
              <span>Higher Threshold $\rightarrow$ Fewer False Alerts</span>
            </div>
          </div>

          {/* Resampling Strategy Selector */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300">Resampling Strategy:</label>
            <div className="grid grid-cols-2 gap-2">
              {resamplings.map(r => (
                <button
                  key={r.id}
                  onClick={() => setSelectedResampling(r.id)}
                  className={`p-2 rounded-lg text-xs font-medium border text-left transition-all cursor-pointer ${
                    selectedResampling === r.id
                      ? 'bg-indigo-600/20 border-indigo-500 text-white'
                      : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:bg-slate-800/60'
                  }`}
                >
                  {r.label}
                </button>
              ))}
            </div>
          </div>

          {/* Financial Loss Matrix Box */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-950/40 to-slate-900 border border-emerald-500/20 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-emerald-400 flex items-center space-x-1">
                <DollarSign className="w-4 h-4" />
                <span>Financial Loss Optimization</span>
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 font-semibold">
                {spec.financial.savingsPercent} Saved
              </span>
            </div>

            <div className="text-2xl font-extrabold text-white">
              ${spec.financial.totalFinancialCost.toLocaleString()}{' '}
              <span className="text-xs text-slate-400 font-normal">total loss</span>
            </div>

            <div className="text-xs text-slate-300 space-y-1">
              <div className="flex justify-between">
                <span>Baseline (No ML Fraud Catch):</span>
                <span className="text-rose-400 font-medium">${spec.financial.baselineNoMLCost.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Net Dollars Saved:</span>
                <span className="text-emerald-400 font-semibold">${spec.financial.costSavings.toLocaleString()}</span>
              </div>
            </div>
          </div>

        </div>

        {/* Center & Right Column: Metrics & Curves (2 cols) */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Top Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="glass-card p-3 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 font-medium">Precision</span>
              <div className="text-xl font-bold text-blue-400">{ (spec.precision * 100).toFixed(1) }%</div>
              <span className="text-[10px] text-slate-500">TP / (TP + FP)</span>
            </div>
            
            <div className="glass-card p-3 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 font-medium">Recall (Sensitivity)</span>
              <div className="text-xl font-bold text-rose-400">{ (spec.recall * 100).toFixed(1) }%</div>
              <span className="text-[10px] text-slate-500">TP / (TP + FN)</span>
            </div>

            <div className="glass-card p-3 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 font-medium">F2 Score ($\beta=2$)</span>
              <div className="text-xl font-bold text-purple-400">{ spec.f2Score }</div>
              <span className="text-[10px] text-slate-500">Recall Weighted</span>
            </div>

            <div className="glass-card p-3 rounded-xl border border-slate-800 text-center">
              <span className="text-[11px] text-slate-400 font-medium">PR-AUC Score</span>
              <div className="text-xl font-bold text-emerald-400">{ spec.prAuc }</div>
              <span className="text-[10px] text-slate-500">Average Precision</span>
            </div>
          </div>

          {/* Confusion Matrix Visualization */}
          <div className="glass-panel p-5 rounded-2xl space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <ShieldAlert className="w-4 h-4 text-indigo-400" />
              <span>Confusion Matrix (56,962 Validation Transactions)</span>
            </h3>

            <div className="grid grid-cols-2 gap-3 max-w-xl mx-auto pt-2">
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-center">
                <span className="text-xs text-emerald-400 font-semibold block">True Positive (TP)</span>
                <div className="text-2xl font-bold text-white my-1">{spec.confusionMatrix.TP}</div>
                <span className="text-[10px] text-slate-400">Frauds Correctly Caught</span>
              </div>

              <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-center">
                <span className="text-xs text-rose-400 font-semibold block">False Negative (FN)</span>
                <div className="text-2xl font-bold text-white my-1">{spec.confusionMatrix.FN}</div>
                <span className="text-[10px] text-slate-400">Missed Frauds ($500 ea)</span>
              </div>

              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-center">
                <span className="text-xs text-amber-400 font-semibold block">False Positive (FP)</span>
                <div className="text-2xl font-bold text-white my-1">{spec.confusionMatrix.FP}</div>
                <span className="text-[10px] text-slate-400">False Alerts ($15 ea)</span>
              </div>

              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 text-center">
                <span className="text-xs text-slate-400 font-semibold block">True Negative (TN)</span>
                <div className="text-2xl font-bold text-white my-1">{spec.confusionMatrix.TN.toLocaleString()}</div>
                <span className="text-[10px] text-slate-400">Legitimate Passed</span>
              </div>
            </div>
          </div>

          {/* Precision-Recall & ROC Curve Tabs/Charts */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <span>Precision-Recall Curve (PR-AUC = {spec.prAuc})</span>
              </h3>
              <span className="text-xs text-slate-400">Critical for Imbalanced Datasets</span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prPoints} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="recall" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'Recall (Sensitivity)', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 10 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                  <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={3} dot={false} name="Model PR Curve" />
                  <Line type="monotone" dataKey="baselineRatio" stroke="#f43f5e" strokeDasharray="5 5" dot={false} name="Random Baseline (0.0017)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
