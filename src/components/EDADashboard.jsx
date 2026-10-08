import React, { useState, useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  AreaChart, Area, CartesianGrid, Legend 
} from 'recharts';
import { FEATURE_PROPERTIES, generateSyntheticDataset } from '../ml/fraudEngine';
import { Info, BarChart2, Clock, DollarSign, Activity } from 'lucide-react';

export default function EDADashboard() {
  const [selectedFeature, setSelectedFeature] = useState('V14');

  // Generate dataset for visual plots
  const rawData = useMemo(() => generateSyntheticDataset(2000, 0.05), []);

  // Compute distribution density for selected feature
  const distributionData = useMemo(() => {
    const prop = FEATURE_PROPERTIES[selectedFeature] || FEATURE_PROPERTIES.V14;
    const bins = 20;
    const minVal = -6;
    const maxVal = 6;
    const step = (maxVal - minVal) / bins;

    const counts = Array.from({ length: bins }, (_, i) => {
      const binStart = minVal + i * step;
      const binEnd = binStart + step;
      return {
        bin: `${binStart.toFixed(1)} to ${binEnd.toFixed(1)}`,
        normal: 0,
        fraud: 0
      };
    });

    rawData.forEach(item => {
      const val = item[selectedFeature];
      if (val !== undefined) {
        const binIdx = Math.min(bins - 1, Math.max(0, Math.floor((val - minVal) / step)));
        if (item.Class === 0) counts[binIdx].normal += 1;
        else counts[binIdx].fraud += 1;
      }
    });

    return counts;
  }, [selectedFeature, rawData]);

  // Compute Hourly Fraud Trends across 48 Hours
  const hourlyData = useMemo(() => {
    const hours = Array.from({ length: 48 }, (_, i) => ({
      hour: `Hr ${i}`,
      normal: 0,
      fraud: 0
    }));

    rawData.forEach(item => {
      const hr = Math.min(47, Math.floor(item.Time / 3600));
      if (item.Class === 0) hours[hr].normal += 1;
      else hours[hr].fraud += 1;
    });

    return hours;
  }, [rawData]);

  // Correlation with Class Target
  const correlations = [
    { feature: 'V14', correlation: -0.303, type: 'Strong Negative' },
    { feature: 'V12', correlation: -0.261, type: 'Negative' },
    { feature: 'V10', correlation: -0.217, type: 'Negative' },
    { feature: 'V17', correlation: -0.326, type: 'Strong Negative' },
    { feature: 'V4',  correlation: 0.134,  type: 'Positive' },
    { feature: 'V11', correlation: 0.155,  type: 'Positive' },
    { feature: 'V2',  correlation: 0.091,  type: 'Weak Positive' },
    { feature: 'Amount', correlation: 0.006, type: 'Near Zero' }
  ];

  return (
    <div className="space-y-6">
      
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel p-5 rounded-2xl">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center space-x-2">
            <BarChart2 className="w-5 h-5 text-indigo-400" />
            <span>Exploratory Data Analysis (EDA)</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Analyzing PCA feature separability, transaction amounts, and temporal fraud peaks in the Kaggle dataset schema.
          </p>
        </div>

        {/* Feature Selector Dropdown */}
        <div className="flex items-center space-x-2 bg-slate-900/90 border border-slate-800 p-1.5 rounded-xl">
          <span className="text-xs font-semibold text-slate-400 px-2">Select PCA Feature:</span>
          <select
            value={selectedFeature}
            onChange={(e) => setSelectedFeature(e.target.value)}
            className="bg-slate-800 text-white text-xs font-semibold px-3 py-1.5 rounded-lg border border-slate-700 outline-none focus:border-indigo-500 cursor-pointer"
          >
            {Object.keys(FEATURE_PROPERTIES).map(key => (
              <option key={key} value={key}>{FEATURE_PROPERTIES[key].name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Feature Distribution & Correlation Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Distribution Chart (2 cols) */}
        <div className="lg:col-span-2 glass-panel p-5 rounded-2xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Activity className="w-4 h-4 text-rose-400" />
                <span>Density Distribution: {FEATURE_PROPERTIES[selectedFeature]?.name}</span>
              </h3>
              <p className="text-xs text-slate-400">
                {FEATURE_PROPERTIES[selectedFeature]?.desc} — Notice clear mean separation between Normal vs Fraud.
              </p>
            </div>
            <div className="flex items-center space-x-3 text-xs">
              <span className="flex items-center space-x-1.5">
                <span className="w-3 h-3 rounded bg-indigo-500"></span>
                <span className="text-slate-300">Normal (Class 0)</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="w-3 h-3 rounded bg-rose-500"></span>
                <span className="text-slate-300">Fraud (Class 1)</span>
              </span>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bin" stroke="#64748b" tick={{ fontSize: 10 }} interval={2} angle={-15} textAnchor="end" />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="normal" fill="#6366f1" radius={[4, 4, 0, 0]} opacity={0.8} name="Normal Count" />
                <Bar dataKey="fraud" fill="#f43f5e" radius={[4, 4, 0, 0]} name="Fraud Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Correlation Table (1 col) */}
        <div className="glass-panel p-5 rounded-2xl space-y-4">
          <div className="border-b border-slate-800/80 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Info className="w-4 h-4 text-emerald-400" />
              <span>Target Class Correlations</span>
            </h3>
            <p className="text-xs text-slate-400">Pearson Correlation coefficient with Fraud Target</p>
          </div>

          <div className="space-y-2 overflow-y-auto max-h-72 pr-1">
            {correlations.map((item, idx) => (
              <div 
                key={idx}
                onClick={() => setSelectedFeature(item.feature)}
                className={`flex items-center justify-between p-2.5 rounded-xl border text-xs cursor-pointer transition-all ${
                  selectedFeature === item.feature
                    ? 'bg-indigo-600/20 border-indigo-500/50 text-white'
                    : 'bg-slate-900/50 border-slate-800/80 text-slate-300 hover:bg-slate-800/60'
                }`}
              >
                <div className="font-semibold">{item.feature}</div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                    item.correlation < 0 
                      ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' 
                      : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  }`}>
                    {item.correlation > 0 ? `+${item.correlation}` : item.correlation}
                  </span>
                  <span className="text-[10px] text-slate-400 hidden sm:inline">{item.type}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Hourly Temporal Fraud Trend */}
      <div className="glass-panel p-5 rounded-2xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Clock className="w-4 h-4 text-indigo-400" />
              <span>Temporal Fraud Pattern across 48 Hours</span>
            </h3>
            <p className="text-xs text-slate-400">
              Notice that fraudulent transactions exhibit distinct spikes during late-night hours when human monitoring is low.
            </p>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={hourlyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="hour" stroke="#64748b" tick={{ fontSize: 10 }} interval={3} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
              />
              <Area type="monotone" dataKey="normal" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} name="Normal Volume" />
              <Area type="monotone" dataKey="fraud" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.4} name="Fraud Spikes" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
