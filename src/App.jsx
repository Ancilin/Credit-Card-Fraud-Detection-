import React, { useState } from 'react';
import Navbar from './components/Navbar';
import KPICards from './components/KPICards';
import EDADashboard from './components/EDADashboard';
import ImbalanceLab from './components/ImbalanceLab';
import ModelStudio from './components/ModelStudio';
import LiveSimulator from './components/LiveSimulator';
import CSVImporter from './components/CSVImporter';
import InterviewGuide from './components/InterviewGuide';
import PythonCodeViewer from './components/PythonCodeViewer';
import { ShieldAlert, Sparkles } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* Navbar Header */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {activeTab === 'overview' && (
          <div className="space-y-8">
            <KPICards setActiveTab={setActiveTab} />
            <EDADashboard />
          </div>
        )}

        {activeTab === 'eda' && (
          <EDADashboard />
        )}

        {activeTab === 'imbalance' && (
          <ImbalanceLab />
        )}

        {activeTab === 'models' && (
          <ModelStudio />
        )}

        {activeTab === 'simulator' && (
          <LiveSimulator />
        )}

        {activeTab === 'csv' && (
          <CSVImporter />
        )}

        {activeTab === 'interview' && (
          <InterviewGuide />
        )}

        {activeTab === 'python' && (
          <PythonCodeViewer />
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-950/80 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-1.5 bg-gradient-to-tr from-rose-600 to-indigo-600 rounded-lg">
              <ShieldAlert className="w-4 h-4 text-white" />
            </div>
            <span className="text-xs font-semibold text-slate-400">
              Credit Card Fraud Detection ML Showcase • Imbalanced Datasets & Classification
            </span>
          </div>

          <div className="flex items-center space-x-4 text-xs text-slate-400 font-medium">
            <span className="flex items-center space-x-1">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              <span>CSV Batch Fraud Engine</span>
            </span>
            <span>•</span>
            <span className="text-slate-500">ML Interview Portfolio Ready</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
