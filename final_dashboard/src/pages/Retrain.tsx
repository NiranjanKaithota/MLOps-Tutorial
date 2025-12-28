import React from 'react';
import { AlertTriangle, Database, RefreshCw, Cpu, HardDrive } from 'lucide-react';

export default function Retrain() {
  return (
    <div className="space-y-8">
      {/* 1. System Interrupt Banner */}
      <div className="bg-amber-500/5 border border-amber-500/20 p-4 rounded-xl flex items-start gap-4">
        <div className="p-2 bg-amber-500/10 rounded-lg text-amber-500">
            <AlertTriangle size={20} />
        </div>
        <div>
          <h4 className="text-amber-500 font-bold font-mono text-sm uppercase tracking-wider">Resource Lock Warning</h4>
          <p className="text-amber-200/60 text-sm mt-1 leading-relaxed">
            Initiating a retraining sequence will lock GPU-0 (RTX 4090) for approximately 18 minutes. 
            Real-time inference will failover to CPU, increasing latency by ~450ms.
          </p>
        </div>
      </div>

      {/* 2. Status Matrix */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800/40 border border-white/5 p-5 rounded-xl backdrop-blur-sm">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-2">Active Node</div>
            <div className="text-white font-mono font-bold text-lg flex items-center gap-2">
                <Cpu size={16} className="text-primary" /> GRU_v2.3
            </div>
        </div>
        <div className="bg-slate-800/40 border border-white/5 p-5 rounded-xl backdrop-blur-sm relative overflow-hidden">
            <div className="absolute inset-0 bg-red-500/5 animate-pulse"></div>
            <div className="text-[10px] text-red-400 uppercase font-bold tracking-widest mb-2 relative z-10">Drift Status</div>
            <div className="text-red-400 font-mono font-bold text-lg relative z-10">
                CRITICAL (1.0)
            </div>
        </div>
        <div className="bg-slate-800/40 border border-white/5 p-5 rounded-xl backdrop-blur-sm">
            <div className="text-[10px] text-slate-500 uppercase font-bold tracking-widest mb-2">Last Checkpoint</div>
            <div className="text-slate-300 font-mono font-bold text-lg flex items-center gap-2">
                <HardDrive size={16} className="text-slate-500" /> 48h ago
            </div>
        </div>
      </div>

      {/* 3. Command Console */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Safe Option */}
        <div className="bg-slate-800/20 p-8 rounded-2xl border border-white/5 hover:border-primary/50 transition-all group relative overflow-hidden">
            <div className="absolute top-0 right-0 p-32 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16 transition-opacity opacity-0 group-hover:opacity-100"></div>
            
            <div className="relative z-10">
                <div className="w-12 h-12 bg-slate-900 rounded-xl border border-white/10 flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform">
                    <Database size={24} />
                </div>
                <h3 className="text-xl font-bold text-white">Conditional Retrain</h3>
                {/* FIX: Used &gt; instead of > to prevent JSX error */}
                <p className="text-slate-400 text-sm mt-2 mb-8 h-10">
                    Executes pipeline `etl_train_v2` only if drift score &gt; 0.5. Uses cached engineered features.
                </p>
                <button className="w-full py-3 bg-primary hover:bg-primary/90 text-slate-900 font-bold rounded-lg transition-colors flex items-center justify-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-slate-900 animate-pulse"></span>
                    EXECUTE SEQUENCE
                </button>
            </div>
        </div>

        {/* Danger Option */}
        <div className="bg-slate-800/20 p-8 rounded-2xl border border-white/5 hover:border-red-500/50 transition-all group relative overflow-hidden">
            <div className="absolute top-0 right-0 p-32 bg-red-500/5 rounded-full blur-3xl -mr-16 -mt-16 transition-opacity opacity-0 group-hover:opacity-100"></div>
            
            <div className="relative z-10">
                <div className="w-12 h-12 bg-slate-900 rounded-xl border border-white/10 flex items-center justify-center text-red-500 mb-6 group-hover:scale-110 transition-transform">
                    <RefreshCw size={24} />
                </div>
                <h3 className="text-xl font-bold text-white">Force Full Pipeline</h3>
                <p className="text-slate-400 text-sm mt-2 mb-8 h-10">
                    Bypasses all checks. Re-downloads raw data, re-generates features, and trains from scratch.
                </p>
                <button className="w-full py-3 bg-transparent border border-red-500/50 text-red-500 hover:bg-red-500 hover:text-white font-bold rounded-lg transition-colors">
                    FORCE OVERRIDE
                </button>
            </div>
        </div>
      </div>
      
      <div className="text-center font-mono text-[10px] text-slate-600">
        ORCHESTRATOR: PREFECT // TRACKING: MLFLOW // VERSIONING: DVC
      </div>
    </div>
  );
}