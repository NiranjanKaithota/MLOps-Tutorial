import React, { useState } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip as ReTooltip, 
  ResponsiveContainer, ReferenceLine, CartesianGrid 
} from 'recharts';
import { AlertOctagon, GitCommit, Search, Sliders, Zap, ExternalLink } from 'lucide-react';

// ... (Keep existing Mock Data Generators: driftTrendData, featureDriftScores) ...
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

export default function DriftMonitoring() {
  const [selectedFeature, setSelectedFeature] = useState("TP2 (Pressure)");

  const openFullReport = () => {
    window.open('http://localhost:8000/static/data_drift_report.html', '_blank');
  };

  return (
    <div className="space-y-8">
      
      {/* 1. Header & Actions */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="md:col-span-3 bg-slate-800/40 backdrop-blur-md p-6 rounded-2xl border border-white/5 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white mb-2">Dataset Drift Status</h2>
            <div className="flex items-center gap-4 text-sm text-slate-400">
               <span className="flex items-center gap-2"><GitCommit size={14}/> Ref: Training Data (v2.3)</span>
               <span className="flex items-center gap-2"><Zap size={14}/> Curr: Live Stream (Last 180)</span>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
              <button 
                onClick={openFullReport}
                className="flex items-center gap-2 px-4 py-2 bg-primary/10 text-primary border border-primary/20 rounded-lg hover:bg-primary/20 transition-colors"
              >
                <ExternalLink size={16} /> Open Evidently Report
              </button>
              
              <div className="text-right pl-6 border-l border-white/10">
                <div className="text-3xl font-mono font-bold text-rose-500">CRITICAL</div>
                <div className="text-xs text-slate-500 uppercase font-bold tracking-widest">Global Drift Score</div>
              </div>
          </div>
        </div>
        
        {/* KPI Card */}
        <div className="bg-rose-500/10 border border-rose-500/20 p-6 rounded-2xl flex flex-col justify-center items-center text-center">
            <AlertOctagon size={32} className="text-rose-500 mb-2" />
            <div className="text-2xl font-bold text-white">2 / 14</div>
            <div className="text-xs text-rose-400 uppercase font-bold">Features Drifted</div>
        </div>
      </div>

      {/* 2. Embedded Report Preview (Iframe) */}
      <div className="bg-white rounded-2xl overflow-hidden border border-white/5 h-[600px] relative group">
          <div className="absolute top-0 left-0 w-full bg-slate-900/80 backdrop-blur-sm p-2 flex justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
              <span className="text-xs text-white font-mono">LIVE PREVIEW: data_drift_report.html</span>
          </div>
          <iframe 
            src="http://localhost:8000/static/data_drift_report.html" 
            className="w-full h-full bg-white"
            title="Evidently Drift Report"
          />
      </div>

    </div>
  );
}