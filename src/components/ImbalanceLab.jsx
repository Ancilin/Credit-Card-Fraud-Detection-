import React, { useState, useMemo } from 'react';
import { simulateResampling, generateSyntheticDataset, getModelPerformanceSpecs } from '../ml/fraudEngine';
import { Layers, AlertCircle, CheckCircle, RefreshCw, Sparkles, HelpCircle } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function ImbalanceLab() {
  const [activeMethod, setActiveMethod] = useState('smote');

  // Generate dataset
  const baseDataset = useMemo(() => generateSyntheticDataset(1500, 0.02), []);

  // Compute resampling output
  const resamplingInfo = useMemo(() => simulateResampling(baseDataset, activeMethod), [baseDataset, activeMethod]);

  // Compute model metrics for selected resampling method
  const modelMetrics = useMemo(() => getModelPerformanceSpecs('xgboost', activeMethod, 0.5), [activeMethod]);

  const methods = [
    { id: 'raw', label: '1. Baseline Imbalanced', tag: '0.17% Fraud', color: 'slate' },
    { id: 'smote', label: '2. SMOTE Oversampling', tag: 'Synthetic 50:50', color: 'indigo' },
    { id: 'undersample', label: '3. NearMiss Undersampling', tag: 'Majority Subsample', color: 'rose' },
    { id: 'class_weight', label: '4. Cost Class Weighting', tag: 'Loss Penalty', color: 'emerald' },
  ];

  // Prepare scatter data for 2D PCA feature space visualization (V14 vs V12)
  const scatterData = useMemo(() => {
    const data = resamplingInfo.scatterData || [];
    return data.map(d => ({
      x: parseFloat(d.V14.toFixed(2)),
      y: parseFloat(d.V12.toFixed(2)),
      type: d.Class === 1 ? (d.isSynthetic ? 'Synthetic Fraud (SMOTE)' : 'Original Fraud') : 'Normal',
      isSynthetic: d.isSynthetic || false,
      fill: d.Class === 1 ? (d.isSynthetic ? '#a855f7' : '#f43f5e') : '#6366f1'
    }));
  }, [resamplingInfo]);

  return (
    <div className="space-y-6">
      
      {/* Title Header */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            <span>Class Imbalance & Resampling Playground</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Compare how SMOTE, Undersampling, and Class Weighting reshape feature decision boundaries.
          </p>
        </div>

        {/* Resampling Method Tabs */}
        <div className="flex flex-wrap gap-2">
          {methods.map((m) => (
            <button
              key={m.id}
              onClick={() => setActiveMethod(m.id)}
              className={`px-3 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer flex items-center space-x-2 ${
                activeMethod === m.id
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg shadow-indigo-500/20 border border-indigo-400/40'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <span>{m.label}</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/30 text-indigo-200">{m.tag}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Class Ratio & Scatter Plot Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Method Description & Ratios */}
        <div className="glass-panel p-5 rounded-2xl space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <h3 className="font-bold text-white text-sm">{resamplingInfo.name}</h3>
            </div>
            
            <p className="text-xs text-slate-300 leading-relaxed bg-slate-900/60 p-3 rounded-xl border border-slate-800">
              {resamplingInfo.description}
            </p>

            {/* Class Distribution Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-indigo-400">Normal ({resamplingInfo.normalCount.toLocaleString()})</span>
                <span className="text-rose-400">Fraud ({resamplingInfo.fraudCount.toLocaleString()})</span>
              </div>

              <div className="h-4 w-full bg-slate-900 rounded-full overflow-hidden flex border border-slate-800">
                <div 
                  className="bg-indigo-600 h-full transition-all duration-500" 
                  style={{ width: `${100 - parseFloat(resamplingInfo.ratio)}%` }}
                ></div>
                <div 
                  className="bg-rose-500 h-full transition-all duration-500" 
                  style={{ width: `${resamplingInfo.ratio}` }}
                ></div>
              </div>
              
              <div className="text-right text-[11px] text-slate-400">
                Minority Ratio: <span className="font-bold text-rose-400">{resamplingInfo.ratio}</span>
              </div>
            </div>

            {/* Performance Snapshot */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800">
                <span className="text-[11px] text-slate-400 font-medium">Recall (Sensitivity)</span>
                <div className="text-lg font-bold text-emerald-400">{ (modelMetrics.recall * 100).toFixed(1) }%</div>
                <span className="text-[10px] text-slate-500">Uncaught Fraud Avoided</span>
              </div>
              <div className="p-3 bg-slate-900/80 rounded-xl border border-slate-800">
                <span className="text-[11px] text-slate-400 font-medium">Precision-Recall AUC</span>
                <div className="text-lg font-bold text-indigo-400">{ modelMetrics.prAuc }</div>
                <span className="text-[10px] text-slate-500">Imbalanced Quality</span>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 flex items-start space-x-2">
            <HelpCircle className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <strong className="block text-indigo-200">Interview Tip:</strong>
              SMOTE interpolates between minority k-nearest neighbors: x_new = x_i + λ(x_zi - x_i) where λ ~ U(0,1).
            </div>
          </div>
        </div>

        {/* Right Column: 2D Feature Space Scatter Plot (2 cols) */}
        <div className="lg:col-span-2 glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <RefreshCw className="w-4 h-4 text-indigo-400" />
                <span>Feature Space Scatter Plot ($V_{14}$ vs $V_{12}$)</span>
              </h3>
              <p className="text-xs text-slate-400">
                Visualizing how resampling alters point density and minority decision boundary resolution.
              </p>
            </div>
            
            <div className="flex items-center space-x-3 text-xs">
              <span className="flex items-center space-x-1">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
                <span className="text-slate-300">Normal</span>
              </span>
              <span className="flex items-center space-x-1">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                <span className="text-slate-300">Real Fraud</span>
              </span>
              {activeMethod === 'smote' && (
                <span className="flex items-center space-x-1">
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-500"></span>
                  <span className="text-slate-300">SMOTE Synthetic</span>
                </span>
              )}
            </div>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" dataKey="x" name="V14" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'V14 PCA', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                <YAxis type="number" dataKey="y" name="V12" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'V12 PCA', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 10 }} />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                />
                <Scatter data={scatterData} fill="#8884d8">
                  {scatterData.map((entry, index) => (
                    <cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

    </div>
  );
}
