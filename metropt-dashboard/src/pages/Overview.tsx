import React from 'react';
import { AlertOctagon, Cpu, ArrowUpRight } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const data = Array.from({ length: 25 }, (_, i) => ({
  time: `T${i}`,
  value: 50 + Math.random() * 30,
  baseline: 40
}));

// Logs with distinct "Badge" styling
const logs = [
  { id: 1, type: "INF", msg: "RUL Prediction: 45h", time: "10:42:15" },
  { id: 2, type: "SYS", msg: "TP2 Sensor Normalized", time: "10:42:05" },
  { id: 3, type: "WRN", msg: "Minor Vibration Spike", time: "10:41:55" },
  { id: 4, type: "INF", msg: "Batch Processed (180)", time: "10:41:45" },
];

export default function Overview() {
  const openDriftReport = () => {
    window.open('http://localhost:8000/static/data_drift_report.html', '_blank');
  };

  return (
    <div className="space-y-8">
      {/* 1. New Alert Style: Gradient Mesh */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-red-500/10 via-orange-500/5 to-background border border-red-500/20 p-6">
        <div className="absolute top-0 right-0 p-3 opacity-10">
            <AlertOctagon size={120} />
        </div>
        <div className="relative z-10 flex justify-between items-start">
            <div>
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <span className="text-red-400">●</span> Drift Threshold Exceeded
                </h3>
                <p className="text-slate-400 mt-2 max-w-2xl">
                    Concept drift detected in incoming sensor streams. The statistical properties of the target variable have shifted significantly (Score: 1.0).
                </p>
            </div>
            {/* <button className="bg-white/5 hover:bg-white/10 text-white border border-white/10 px-6 py-2 rounded-lg font-medium backdrop-blur-sm transition-all">
                Analyze Drift
            </button> */}
            <button 
                onClick={openDriftReport}
                className="bg-white/5 hover:bg-white/10 text-white border border-white/10 px-6 py-2 rounded-lg font-medium backdrop-blur-sm transition-all flex items-center gap-2"
            >
                {/* <Search size={16} /> */}Analyze Drift Report 
            </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* 2. Main Chart - Glass Card */}
        <div className="lg:col-span-8 bg-slate-800/40 backdrop-blur-md p-6 rounded-2xl border border-white/5 shadow-xl">
          <div className="flex justify-between items-center mb-6">
            <div>
                <h3 className="font-bold text-white">RUL Projection</h3>
                <p className="text-xs text-slate-500 font-mono mt-1">MODEL: GRU_V2.3 // WINDOW: 180</p>
            </div>
            {/* Legend */}
            <div className="flex gap-4 text-xs font-mono">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-primary"></span> Predicted</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-600"></span> Baseline</span>
            </div>
          </div>
          
          <div className="h-[320px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} opacity={0.4} />
                <XAxis dataKey="time" stroke="#64748b" tick={{fontSize: 12, fontFamily: 'monospace'}} axisLine={false} tickLine={false} />
                <YAxis stroke="#64748b" tick={{fontSize: 12, fontFamily: 'monospace'}} axisLine={false} tickLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff', fontFamily: 'monospace' }}
                  cursor={{stroke: '#06b6d4', strokeWidth: 1}}
                />
                <Area type="monotone" dataKey="value" stroke="#06b6d4" strokeWidth={3} fillOpacity={1} fill="url(#colorVal)" />
                <Area type="monotone" dataKey="baseline" stroke="#475569" strokeWidth={2} strokeDasharray="4 4" fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 3. Terminal Style Logs */}
        <div className="lg:col-span-4 flex flex-col gap-6">
            {/* KPI Card */}
            <div className="bg-gradient-to-br from-indigo-500/20 to-purple-500/20 p-6 rounded-2xl border border-indigo-500/20 relative overflow-hidden group">
                <div className="absolute right-0 top-0 p-4 opacity-0 group-hover:opacity-20 transition-opacity">
                    <ArrowUpRight size={48} className="text-white" />
                </div>
                <h4 className="text-indigo-300 text-sm font-medium uppercase tracking-wider">Confidence</h4>
                <div className="text-4xl font-bold text-white mt-2 font-mono">98.2<span className="text-lg text-indigo-300">%</span></div>
                <p className="text-xs text-indigo-200/60 mt-2">Running on GPU-01</p>
            </div>

            {/* Log Panel */}
            <div className="bg-slate-800/40 backdrop-blur-md p-0 rounded-2xl border border-white/5 flex-1 overflow-hidden flex flex-col">
                <div className="p-4 border-b border-white/5 bg-slate-900/50 flex justify-between items-center">
                    <h3 className="font-bold text-white text-sm">System Events</h3>
                    <Cpu size={14} className="text-slate-500" />
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
                    {logs.map((log) => (
                    <div key={log.id} className="text-sm flex gap-3 items-start group">
                        <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                            log.type === 'WRN' ? 'border-orange-500/30 text-orange-400 bg-orange-500/10' : 
                            'border-primary/30 text-primary bg-primary/10'
                        }`}>
                            {log.type}
                        </span>
                        <div className="flex-1">
                            <span className="text-slate-300 block text-xs group-hover:text-white transition-colors">{log.msg}</span>
                            <span className="text-[10px] text-slate-600 font-mono">{log.time}</span>
                        </div>
                    </div>
                    ))}
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}