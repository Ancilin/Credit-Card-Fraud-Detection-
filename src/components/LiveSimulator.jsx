import React, { useState, useMemo } from 'react';
import { 
  predictFraudScore, 
  PRESET_TRANSACTIONS, 
  FEATURE_PROPERTIES 
} from '../ml/fraudEngine';
import { Zap, ShieldCheck, ShieldAlert, AlertTriangle, RefreshCw, BarChart2 } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function LiveSimulator() {
  const [features, setFeatures] = useState(PRESET_TRANSACTIONS[0].features);

  // Compute live prediction
  const prediction = useMemo(() => predictFraudScore(features), [features]);

  const handlePresetSelect = (preset) => {
    setFeatures({ ...preset.features });
    if (preset.name.includes('Suspicious') || preset.name.includes('Bot')) {
      confetti({ particleCount: 30, spread: 60, origin: { y: 0.6 } });
    }
  };

  const handleSliderChange = (key, value) => {
    setFeatures(prev => ({
      ...prev,
      [key]: parseFloat(value)
    }));
  };

  const badgeStyles = {
    rose: 'bg-rose-500/10 text-rose-400 border-rose-500/30 animate-pulse',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
  };

  return (
    <div className="space-y-6">
      
      {/* Title Header */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <Zap className="w-5 h-5 text-rose-400 animate-pulse" />
            <span>Real-Time Fraud Detection & Transaction Simulator</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Test live transaction risk scoring with custom feature values or preset fraud scenarios.
          </p>
        </div>

        {/* Quick Presets */}
        <div className="flex flex-wrap gap-2">
          {PRESET_TRANSACTIONS.map((preset, idx) => (
            <button
              key={idx}
              onClick={() => handlePresetSelect(preset)}
              className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-900/80 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800 transition-all cursor-pointer"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Controls vs Prediction Output */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Feature Controls (2 cols) */}
        <div className="lg:col-span-2 glass-panel p-5 rounded-2xl space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <RefreshCw className="w-4 h-4 text-indigo-400" />
              <span>Transaction Feature Inputs ($V_1-V_{28}$, Amount, Time)</span>
            </h3>
            <button
              onClick={() => setFeatures(PRESET_TRANSACTIONS[0].features)}
              className="text-xs text-indigo-400 hover:underline font-semibold cursor-pointer"
            >
              Reset to Safe Normal
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.keys(FEATURE_PROPERTIES).map((key) => {
              const prop = FEATURE_PROPERTIES[key];
              const val = features[key] !== undefined ? features[key] : prop.normalMean;
              const isAmount = key === 'Amount';

              return (
                <div key={key} className="space-y-1.5 p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                  <div className="flex justify-between text-xs">
                    <span className="font-semibold text-slate-300">{prop.name}</span>
                    <span className="font-bold text-indigo-400">
                      {isAmount ? `$${val.toFixed(2)}` : val.toFixed(2)}
                    </span>
                  </div>

                  <input
                    type="range"
                    min={isAmount ? 1 : -6}
                    max={isAmount ? 3500 : 6}
                    step={isAmount ? 5 : 0.1}
                    value={val}
                    onChange={(e) => handleSliderChange(key, e.target.value)}
                    className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
                  />

                  <div className="text-[10px] text-slate-500 truncate">{prop.desc}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Live Risk Output & SHAP Waterfall (1 col) */}
        <div className="glass-panel p-5 rounded-2xl space-y-6 flex flex-col justify-between">
          
          {/* Risk Card Banner */}
          <div className="space-y-4">
            <div className="text-center p-6 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 space-y-3 shadow-xl relative overflow-hidden">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Calculated Fraud Probability</span>
              
              <div className="text-4xl font-extrabold text-white tracking-tight">
                {prediction.percentScore}
              </div>

              <div className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold border ${badgeStyles[prediction.badgeColor]}`}>
                {prediction.badgeColor === 'rose' ? <ShieldAlert className="w-4 h-4" /> : <ShieldCheck className="w-4 h-4" />}
                <span>{prediction.riskLevel}</span>
              </div>
            </div>

            {/* SHAP Feature Impact Waterfall */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-300 flex items-center space-x-1.5">
                <BarChart2 className="w-3.5 h-3.5 text-indigo-400" />
                <span>SHAP Feature Impact Waterfall</span>
              </h4>

              <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {prediction.attributions.map((attr, idx) => {
                  const isPositiveRisk = attr.contribution > 0;
                  return (
                    <div key={idx} className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800 text-xs">
                      <div className="truncate max-w-[120px]">
                        <span className="font-semibold text-slate-200 block truncate">{attr.feature}</span>
                        <span className="text-[10px] text-slate-500">val: {attr.value.toFixed(2)}</span>
                      </div>

                      <div className="flex items-center space-x-2">
                        <div className="w-16 h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`h-full ${isPositiveRisk ? 'bg-rose-500' : 'bg-emerald-500'}`}
                            style={{ width: `${Math.min(100, Math.abs(attr.contribution) * 40)}%` }}
                          ></div>
                        </div>

                        <span className={`font-mono text-xs font-bold w-12 text-right ${isPositiveRisk ? 'text-rose-400' : 'text-emerald-400'}`}>
                          {isPositiveRisk ? `+${attr.contribution}` : attr.contribution}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>

          <div className="text-[11px] text-slate-400 bg-slate-900/80 p-3 rounded-xl border border-slate-800">
            <strong>Model Threshold Policy:</strong> Transactions exceeding <span className="text-amber-400 font-bold">40%</span> prompt SMS 2FA step-up; transactions over <span className="text-rose-400 font-bold">75%</span> trigger immediate card hold.
          </div>

        </div>

      </div>

    </div>
  );
}
