import React, { useState, useRef } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip as ReTooltip, 
  ResponsiveContainer, ReferenceLine, CartesianGrid 
} from 'recharts';
import { AlertOctagon, GitCommit, Search, Sliders, Zap, ExternalLink, RefreshCw, FileText } from 'lucide-react';

// --- MOCK DATA FOR VISUALIZATION (Trends usually come from a database) ---
const driftTrendData = Array.from({ length: 30 }, (_, i) => ({
  day: `Nov ${i + 1}`,
  score: Math.min(1.0, Math.max(0, (i < 20 ? 0.1 : 0.1 + (i - 20) * 0.05) + Math.random() * 0.1)),
  threshold: 0.5
}));

const featureDriftScores = [
  { feature: "TP2 (Pressure)", score: 0.85, status: "DRIFT DETECTED" },
  { feature: "H1 (Heat)", score: 0.62, status: "DRIFT DETECTED" },
  { feature: "Motor_current", score: 0.45, status: "Stable" },
  { feature: "Oil_temperature", score: 0.38, status: "Stable" },
  { feature: "Vibration", score: 0.12, status: "Stable" },
];

const REPORT_URL = "http://localhost:8000/static/data_drift_report.html";

export default function DriftMonitoring() {
  const [selectedFeature, setSelectedFeature] = useState("TP2 (Pressure)");
  const [iframeKey, setIframeKey] = useState(0); // Used to force iframe refresh
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    setIframeKey(prev => prev + 1); // Remounts/Reloads iframe
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  const openFullReport = () => {
    window.open(REPORT_URL, '_blank');
  };

  return (
    <div className="space-y-8">
      
      {/* 1. Header & Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="md:col-span-3 bg-slate-800/40 backdrop-blur-md p-6 rounded-2xl border border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
                <FileText className="text-primary" size={24}/> 
                Dataset Drift Status
            </h2>
            <div className="flex items-center gap-4 text-sm text-slate-400 font-mono">
               <span className="flex items-center gap-2 px-2 py-1 bg-slate-800 rounded border border-white/5">
                   <GitCommit size={14} className="text-indigo-400"/> Ref: Training (v2.3)
               </span>
               <span className="flex items-center gap-2 px-2 py-1 bg-slate-800 rounded border border-white/5">
                   <Zap size={14} className="text-emerald-400"/> Curr: Live Window (180)
               </span>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
              <button 
                onClick={handleRefresh}
                disabled={isRefreshing}
                className={`p-2 rounded-lg border border-white/10 text-slate-400 hover:text-white hover:bg-white/5 transition-all ${isRefreshing ? 'animate-spin text-primary' : ''}`}
                title="Refresh Report"
              >
                <RefreshCw size={20} />
              </button>

              <button 
                onClick={openFullReport}
                className="flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary border border-primary/20 rounded-lg hover:bg-primary/20 transition-colors font-medium text-sm"
              >
                <ExternalLink size={16} /> Open Full Report
              </button>
              
              <div className="text-right pl-6 border-l border-white/10 hidden md:block">
                <div className="text-3xl font-mono font-bold text-rose-500">CRITICAL</div>
                <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Global Drift Score</div>
              </div>
          </div>
        </div>
        
        {/* KPI Card */}
        <div className="bg-rose-500/10 border border-rose-500/20 p-6 rounded-2xl flex flex-col justify-center items-center text-center shadow-[0_0_20px_rgba(244,63,94,0.1)]">
            <AlertOctagon size={32} className="text-rose-500 mb-2" />
            <div className="text-3xl font-bold text-white font-mono">2 / 5</div>
            <div className="text-xs text-rose-400 uppercase font-bold tracking-wider mt-1">Features Drifted</div>
        </div>
      </div>

      {/* 2. Embedded Report Preview (Iframe) */}
      <div className="bg-white rounded-2xl overflow-hidden border border-white/5 h-[800px] relative group shadow-2xl">
          {/* Header Overlay */}
          <div className="absolute top-0 left-0 w-full h-10 bg-slate-900 flex items-center px-4 justify-between border-b border-white/10 z-10">
              <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-red-500/50"/>
                  <div className="w-3 h-3 rounded-full bg-amber-500/50"/>
                  <div className="w-3 h-3 rounded-full bg-emerald-500/50"/>
              </div>
              <span className="text-xs text-slate-400 font-mono flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                  {REPORT_URL}
              </span>
          </div>

          {/* The Report Iframe */}
          <iframe 
            key={iframeKey} // Changing this key forces React to reload the iframe
            src={REPORT_URL} 
            className="w-full h-full bg-slate-50 pt-10" // pt-10 to account for the custom header
            title="Evidently Drift Report"
            // Sandbox permissions to allow scripts inside the report but prevent top-level navigation
            sandbox="allow-same-origin allow-scripts allow-popups allow-forms"
          />
          
          {/* Overlay for "Not Found" handling could be added here if needed, 
              but usually the browser default 404 is clear enough for dev tools */}
      </div>

    </div>
  );
}