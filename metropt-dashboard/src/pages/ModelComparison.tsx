import React from 'react';
import { Trophy, GitCommit, Activity, Server } from 'lucide-react';

export default function ModelComparison() {
  // Updated with your latest training results
  const models = [
    { 
      name: "LSTM_v2.4", 
      type: "Recurrent", 
      acc: "87.0%", 
      mae: "08.73h", // Kept previous MAE; update if you have new specific MAE data
      status: "ONLINE", 
      latency: "28ms",
      isChampion: true 
    },
    { 
      name: "GRU_v2.3", 
      type: "Recurrent", 
      acc: "69.0%", 
      mae: "08.99h", 
      status: "STANDBY", 
      latency: "24ms",
      isChampion: false 
    },
    { 
      name: "CNN_v1.0", 
      type: "Convolutional", 
      acc: "60.0%", 
      mae: "11.10h", 
      status: "STANDBY", 
      latency: "18ms",
      isChampion: false 
    },
  ];

  // Dynamically find the champion model (Production)
  const champion = models.find(m => m.isChampion) || models[0];

  return (
    <div className="space-y-8">
      {/* 1. Champion Node Hero (Dynamically uses LSTM now) */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-primary/20 via-slate-900 to-slate-900 border border-primary/20 p-8 shadow-glow">
        <div className="absolute top-0 right-0 p-8 opacity-10">
          <Server size={200} className="text-primary" />
        </div>
        
        <div className="relative z-10 flex items-center gap-8">
          <div className="w-24 h-24 bg-slate-900/50 rounded-2xl border border-primary/30 flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.3)]">
            <Trophy size={40} className="text-primary" />
          </div>
          
          <div>
            <div className="flex items-center gap-3 mb-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-primary text-slate-900 uppercase tracking-wider">Production</span>
                <span className="text-slate-400 text-sm font-mono flex items-center gap-2">
                    <GitCommit size={14} /> sha256:7b2...9a1
                </span>
            </div>
            {/* Dynamic Title */}
            <h1 className="text-4xl font-bold text-white tracking-tight">{champion.name}</h1>
            <div className="flex items-center gap-6 mt-4">
                <div>
                    <div className="text-xs text-slate-500 uppercase font-bold">Accuracy</div>
                    {/* Dynamic Accuracy */}
                    <div className="text-2xl text-emerald-400 font-mono font-bold">{champion.acc}</div>
                </div>
                <div className="w-px h-8 bg-white/10"></div>
                <div>
                    <div className="text-xs text-slate-500 uppercase font-bold">Mean Error</div>
                    {/* Dynamic MAE */}
                    <div className="text-2xl text-primary font-mono font-bold">±{champion.mae}</div>
                </div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Technical Comparison Table */}
      <div className="bg-slate-800/40 backdrop-blur-md rounded-2xl border border-white/5 overflow-hidden">
        <div className="p-6 border-b border-white/5 flex items-center gap-3">
            <Activity size={18} className="text-slate-400" />
            <h3 className="font-bold text-white text-sm uppercase tracking-wide">Model Registry</h3>
        </div>
        
        <table className="w-full text-left">
          <thead className="bg-slate-900/50 text-xs text-slate-500 uppercase font-mono">
            <tr>
              <th className="px-6 py-4">Architecture</th>
              <th className="px-6 py-4">Type</th>
              <th className="px-6 py-4">Accuracy</th>
              <th className="px-6 py-4">MAE Score</th>
              <th className="px-6 py-4">Latency</th>
              <th className="px-6 py-4 text-right">State</th>
            </tr>
          </thead>
          <tbody className="text-sm divide-y divide-white/5">
            {models.map((m) => (
              <tr key={m.name} className="hover:bg-white/5 transition-colors group">
                <td className="px-6 py-4 font-bold text-white group-hover:text-primary transition-colors">{m.name}</td>
                <td className="px-6 py-4 text-slate-400 font-mono text-xs">{m.type}</td>
                <td className={`px-6 py-4 font-mono ${parseInt(m.acc) >= 80 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {m.acc}
                </td>
                <td className="px-6 py-4 text-slate-300 font-mono">{m.mae}</td>
                <td className="px-6 py-4 text-slate-500 font-mono">{m.latency}</td>
                <td className="px-6 py-4 text-right">
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-bold tracking-wide border ${
                        m.status === 'ONLINE' ? 'border-primary/30 bg-primary/10 text-primary shadow-[0_0_10px_rgba(6,182,212,0.2)]' : 
                        m.status === 'STANDBY' ? 'border-amber-500/30 bg-amber-500/10 text-amber-400' :
                        'border-slate-700 bg-slate-800 text-slate-500'
                    }`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${
                            m.status === 'ONLINE' ? 'bg-primary animate-pulse' : 
                            m.status === 'STANDBY' ? 'bg-amber-400' : 'bg-slate-600'
                        }`}></span>
                        {m.status}
                    </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}