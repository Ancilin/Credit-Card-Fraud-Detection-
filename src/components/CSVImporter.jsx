import React, { useState } from 'react';
import { 
  parseCSVAndPredictBatch, 
  generateSampleCSVTemplate 
} from '../ml/fraudEngine';
import { 
  UploadCloud, 
  FileSpreadsheet, 
  Download, 
  ShieldAlert, 
  ShieldCheck, 
  AlertTriangle, 
  CheckCircle2, 
  Search, 
  Filter, 
  ArrowUpDown,
  RefreshCw,
  Sparkles
} from 'lucide-react';

export default function CSVImporter() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [error, setError] = useState('');
  const [filterRisk, setFilterRisk] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    if (!uploadedFile.name.endsWith('.csv')) {
      setError('Please upload a valid .csv file.');
      return;
    }

    setError('');
    setFile(uploadedFile);
    setLoading(true);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const results = parseCSVAndPredictBatch(text);
        setBatchResult(results);
        setLoading(false);
      } catch (err) {
        setError(err.message || 'Error parsing CSV file.');
        setLoading(false);
      }
    };
    reader.onerror = () => {
      setError('Failed to read CSV file.');
      setLoading(false);
    };
    reader.readAsText(uploadedFile);
  };

  const handleDownloadSample = () => {
    const csvContent = generateSampleCSVTemplate();
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'sample_credit_card_fraud_test.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportPredictionsCSV = () => {
    if (!batchResult || !batchResult.predictions) return;

    const headers = ['Row', 'Transaction_ID', 'Time', 'Amount', 'Fraud_Probability', 'Risk_Level', 'Primary_Risk_Marker'];
    const rows = batchResult.predictions.map(p => [
      p.rowNumber,
      `"${p.id}"`,
      p.Time,
      p.Amount,
      p.probability,
      `"${p.riskLevel}"`,
      `"${p.topRiskFeature}"`
    ]);

    const csvText = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvText], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `fraud_predictions_${file ? file.name : 'batch'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filter prediction results
  const filteredPredictions = (batchResult?.predictions || []).filter(item => {
    const matchesSearch = item.id.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          item.topRiskFeature.toLowerCase().includes(searchQuery.toLowerCase());
    
    if (filterRisk === 'ALL') return matchesSearch;
    if (filterRisk === 'FRAUD') return matchesSearch && item.riskLevel === 'CRITICAL FRAUD';
    if (filterRisk === 'SUSPICIOUS') return matchesSearch && (item.riskLevel === 'HIGH SUSPICION' || item.riskLevel === 'MODERATE RISK');
    if (filterRisk === 'SAFE') return matchesSearch && item.riskLevel === 'LOW';
    return matchesSearch;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold mb-2">
            <UploadCloud className="w-3.5 h-3.5" />
            <span>CSV Batch File Detection Engine</span>
          </div>
          <h2 className="text-xl font-bold text-white">Import External Credit Card Dataset</h2>
          <p className="text-xs text-slate-400 mt-1">
            Upload your own transaction `.csv` file (e.g. Kaggle `creditcard.csv`) to run instant batch fraud prediction and export annotated results.
          </p>
        </div>

        <button
          onClick={handleDownloadSample}
          className="inline-flex items-center space-x-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition-all cursor-pointer"
        >
          <Download className="w-4 h-4 text-indigo-400" />
          <span>Download Sample CSV Template</span>
        </button>
      </div>

      {/* Upload Zone */}
      <div className="glass-panel p-8 rounded-2xl border-2 border-dashed border-slate-800 hover:border-indigo-500/50 transition-all text-center space-y-4 relative overflow-hidden group">
        <input
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
        />

        <div className="w-14 h-14 mx-auto rounded-2xl bg-indigo-600/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
          <FileSpreadsheet className="w-7 h-7" />
        </div>

        <div className="space-y-1">
          <h3 className="text-base font-bold text-white">
            {file ? file.name : 'Click to Upload or Drag & Drop Credit Card CSV'}
          </h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Supports standard Kaggle schema with headers `Time`, `Amount`, `V1`, `V2`... `V28` or arbitrary transaction CSVs.
          </p>
        </div>

        {file && (
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-indigo-300">
            <span>File size: {(file.size / 1024).toFixed(1)} KB</span>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center space-x-2 text-indigo-400 text-xs font-semibold animate-pulse">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span>Analyzing dataset transactions and computing risk probabilities...</span>
          </div>
        )}

        {error && (
          <div className="text-xs text-rose-400 bg-rose-500/10 p-2.5 rounded-xl border border-rose-500/20 max-w-md mx-auto font-medium">
            {error}
          </div>
        )}
      </div>

      {/* Batch Results View */}
      {batchResult && (
        <div className="space-y-6">
          
          {/* Summary Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <span className="text-xs text-slate-400 font-medium">Total Processed</span>
              <div className="text-2xl font-bold text-white mt-1">{batchResult.totalProcessed.toLocaleString()}</div>
              <span className="text-[10px] text-slate-500">Transactions evaluated</span>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800 bg-rose-500/5">
              <span className="text-xs text-rose-400 font-semibold flex items-center space-x-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Critical Fraud Alerts</span>
              </span>
              <div className="text-2xl font-bold text-rose-400 mt-1">{batchResult.totalFraud}</div>
              <span className="text-[10px] text-slate-400">High severity risk</span>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800 bg-amber-500/5">
              <span className="text-xs text-amber-400 font-semibold flex items-center space-x-1">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>High / Moderate Suspicion</span>
              </span>
              <div className="text-2xl font-bold text-amber-400 mt-1">{batchResult.totalHighSuspicion + batchResult.totalModerate}</div>
              <span className="text-[10px] text-slate-400">Step-up 2FA required</span>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800 bg-emerald-500/5">
              <span className="text-xs text-emerald-400 font-semibold flex items-center space-x-1">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span>Fraud Dollar Risk</span>
              </span>
              <div className="text-2xl font-bold text-white mt-1">${batchResult.totalDollarAtRisk.toLocaleString()}</div>
              <span className="text-[10px] text-emerald-400 font-medium">Total dollar value at risk</span>
            </div>

          </div>

          {/* Results Table Header & Search */}
          <div className="glass-panel p-5 rounded-2xl space-y-4">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-indigo-400" />
                  <span>Batch Detection Results Table</span>
                </h3>
                <p className="text-xs text-slate-400">
                  Showing predicted risk levels and probability scores for imported CSV rows.
                </p>
              </div>

              {/* Download CSV Button */}
              <button
                onClick={handleExportPredictionsCSV}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-indigo-600 text-white font-semibold text-xs shadow-lg shadow-emerald-600/20 hover:opacity-90 transition-all cursor-pointer shrink-0"
              >
                <Download className="w-4 h-4" />
                <span>Export Predicted CSV</span>
              </button>
            </div>

            {/* Filter Tabs & Search Bar */}
            <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
              
              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-xl border border-slate-800 w-full sm:w-auto">
                {[
                  { id: 'ALL', label: 'All' },
                  { id: 'FRAUD', label: 'Critical Fraud' },
                  { id: 'SUSPICIOUS', label: 'Suspicious' },
                  { id: 'SAFE', label: 'Legitimate' },
                ].map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setFilterRisk(t.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      filterRisk === t.id
                        ? 'bg-indigo-600 text-white shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Search input */}
              <div className="relative w-full sm:w-64">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Search ID or feature..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 outline-none focus:border-indigo-500"
                />
              </div>

            </div>

            {/* Table Container */}
            <div className="overflow-x-auto border border-slate-800 rounded-xl max-h-[420px]">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-slate-950 text-slate-400 sticky top-0 z-10 border-b border-slate-800">
                  <tr>
                    <th className="p-3 font-semibold">Row</th>
                    <th className="p-3 font-semibold">Transaction ID</th>
                    <th className="p-3 font-semibold">Time (s)</th>
                    <th className="p-3 font-semibold">Amount ($)</th>
                    <th className="p-3 font-semibold">Fraud Risk %</th>
                    <th className="p-3 font-semibold">Risk Level</th>
                    <th className="p-3 font-semibold">Top Risk Marker</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {filteredPredictions.length === 0 ? (
                    <tr>
                      <td colSpan="7" className="p-6 text-center text-slate-500">
                        No transactions found matching the selected filter.
                      </td>
                    </tr>
                  ) : (
                    filteredPredictions.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/60 transition-colors">
                        <td className="p-3 font-mono text-slate-500">{row.rowNumber}</td>
                        <td className="p-3 font-semibold text-white">{row.id}</td>
                        <td className="p-3 text-slate-400">{row.Time}s</td>
                        <td className="p-3 font-bold text-white">${row.Amount.toFixed(2)}</td>
                        <td className="p-3 font-mono font-bold text-indigo-400">{row.percentScore}</td>
                        <td className="p-3">
                          <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold border ${
                            row.riskLevel === 'CRITICAL FRAUD' ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 animate-pulse' :
                            row.riskLevel === 'HIGH SUSPICION' ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' :
                            row.riskLevel === 'MODERATE RISK' ? 'bg-blue-500/10 text-blue-400 border-blue-500/30' :
                            'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          }`}>
                            {row.riskLevel}
                          </span>
                        </td>
                        <td className="p-3 text-slate-400 font-mono text-[11px]">{row.topRiskFeature}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

          </div>

        </div>
      )}

    </div>
  );
}
