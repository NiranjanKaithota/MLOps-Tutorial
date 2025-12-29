import React, { useEffect, useState, useRef } from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { Database, Thermometer, Zap, Activity, Play, Pause, AlertCircle } from 'lucide-react';

// Types
interface PredictionRow {
  id: number;
  time: string;
  gru_rul: string;
  cnn_rul: string;
  lstm_rul: string;
  status: string;
}

interface Telemetry {
  motor_current: number;
  oil_temp: number;
  pressure: number;
  vibration: number;
}

// Metric Card Component
const MetricCard = ({ label, value, unit, icon: Icon, variant = 'cyan' }: any) => {
  const styles = {
    cyan: { iconBg: 'bg-cyan-500/10', iconText: 'text-cyan-400', stroke: '#06b6d4' },
    rose: { iconBg: 'bg-rose-500/10', iconText: 'text-rose-400', stroke: '#f43f5e' }
  };
  const theme = styles[variant as keyof typeof styles] || styles.cyan;

  return (
    <div className="bg-slate-800/40 backdrop-blur-sm p-5 rounded-2xl border border-white/5 hover:border-primary/30 transition-all group">
      <div className="flex justify-between items-start mb-4">
          <div className={`p-2 rounded-lg ${theme.iconBg} ${theme.iconText}`}>
              <Icon size={20} />
          </div>
          <div className="text-right">
              <div className="text-2xl font-bold text-white font-mono">{value}</div>
              <div className="text-xs text-slate-500 uppercase font-medium">{unit}</div>
          </div>
      </div>
      <div className="h-10 w-full opacity-50 group-hover:opacity-100 transition-opacity">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={Array.from({length:10}, () => ({v: Math.random()}))}>
            <Line type="monotone" dataKey="v" stroke={theme.stroke} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-xs text-slate-400 font-medium">{label}</div>
    </div>
  );
};

export default function LivePredictions() {
  const [data, setData] = useState<PredictionRow[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry>({ 
    motor_current: 0, oil_temp: 0, pressure: 0, vibration: 0 
  });
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Use a ref to prevent race conditions in strict mode
  const isFetching = useRef(false);

  // 1. Function to Reset the Simulation
  const handleStartStop = async () => {
    if (!isPlaying) {
        // RESET logic: Tell backend to start from index 0
        try {
            await fetch('/api/reset', { method: 'POST' }); // Relative path
            setData([]); // Clear old table data
            setError(null);
            setIsPlaying(true);
        } catch (err) {
            console.error("Failed to reset:", err);
            setError("Backend not reachable. Check Docker logs.");
        }
    } else {
        setIsPlaying(false);
    }
  };

  // 2. Fetch Data from Backend
  const fetchNextStep = async () => {
    if (isFetching.current) return;
    isFetching.current = true;

    try {
      // Use Relative URL - Nginx/Vite will proxy this to port 8000
      const res = await fetch('/api/next'); 
      
      if (!res.ok) {
          throw new Error(`Server Error: ${res.status}`);
      }

      const json = await res.json();
      
      if (json.status === 'finished') {
        setIsPlaying(false);
        return;
      }

      setTelemetry(json.telemetry || { motor_current: 0, oil_temp: 0, pressure: 0, vibration: 0 });

      const newRow: PredictionRow = {
        id: json.index,
        time: json.timestamp ? json.timestamp.substring(11, 19) : "--:--:--",
        gru_rul: `${(json.predictions?.GRU || 0).toFixed(1)}h`,
        cnn_rul: `${(json.predictions?.CNN || 0).toFixed(1)}h`,
        lstm_rul: `${(json.predictions?.LSTM || 0).toFixed(1)}h`,
        status: json.status || "UNKNOWN"
      };

      setData(prev => [newRow, ...prev].slice(0, 15));
      setError(null);
      
    } catch (err: any) {
      console.error("API Error", err);
      setError("Stream disconnected. Is Backend running?");
      setIsPlaying(false);
    } finally {
        isFetching.current = false;
    }
  };

  useEffect(() => {
    let interval: any;
    if (isPlaying) {
      interval = setInterval(fetchNextStep, 1000); // 1000ms = 1 second
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className="space-y-6">
      
      {/* 1. Control & Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {/* Control Button */}
        <button 
            onClick={handleStartStop}
            className={`md:col-span-1 h-full rounded-2xl font-bold text-xs uppercase tracking-wide flex flex-col items-center justify-center gap-3 transition-all shadow-lg border ${
                isPlaying 
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/20 hover:bg-rose-500/20' 
                : 'bg-primary/10 text-primary border-primary/20 hover:bg-primary/20'
            }`}
        >
            <div className={`p-3 rounded-full ${isPlaying ? 'bg-rose-500 text-white' : 'bg-primary text-white'}`}>
                {isPlaying ? <Pause size={24} /> : <Play size={24} className="ml-1" />}
            </div>
            {isPlaying ? 'Stop Stream' : 'Start Simulation'}
        </button>

        {/* Telemetry Cards */}
        <MetricCard label="Motor Current" value={telemetry.motor_current?.toFixed(2)} unit="AMP" icon={Zap} variant="cyan" />
        <MetricCard label="Oil Temp" value={telemetry.oil_temp?.toFixed(1)} unit="CELSIUS" icon={Thermometer} variant="rose" />
        <MetricCard label="Pressure TP2" value={telemetry.pressure?.toFixed(2)} unit="BAR" icon={Database} variant="cyan" />
        <MetricCard label="Vibration" value={telemetry.vibration?.toFixed(3)} unit="MM/S" icon={Activity} variant="cyan" />
      </div>

      {/* 2. Error Banner (Only shows if there is an error) */}
      {error && (
          <div className="bg-red-500/10 border border-red-500/20 p-4 rounded-xl flex items-center gap-3 text-red-400 text-sm">
              <AlertCircle size={20} />
              {error}
          </div>
      )}

      {/* 3. Data Table */}
      <div className="bg-slate-800/40 backdrop-blur-md rounded-2xl border border-white/5 overflow-hidden flex flex-col min-h-[500px]">
        <div className="p-6 border-b border-white/5 flex justify-between items-center bg-slate-900/20">
            <h3 className="font-bold text-white">Live Model Consensus</h3>
            <div className="flex items-center gap-4">
                <span className="text-xs text-slate-500 font-mono hidden sm:inline-block">Source: Live Backend Stream</span>
                {isPlaying && (
                    <span className="flex items-center gap-2 text-[10px] text-emerald-400 font-bold px-2 py-1 bg-emerald-500/10 rounded-full">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/> LIVE
                    </span>
                )}
            </div>
        </div>
        
        <div className="flex-1 overflow-auto">
            <table className="w-full text-left border-collapse">
            <thead className="bg-slate-900/50 text-xs text-slate-400 uppercase font-medium sticky top-0 z-10 backdrop-blur-sm">
                <tr>
                    <th className="py-4 px-6 text-slate-500">Timestamp</th>
                    <th className="py-4 px-6 text-primary">GRU</th>
                    <th className="py-4 px-6 text-blue-400">CNN</th>
                    <th className="py-4 px-6 text-indigo-400">LSTM</th>
                    <th className="py-4 px-6 text-right">Consensus Status</th>
                </tr>
            </thead>
            <tbody className="text-sm divide-y divide-white/5">
                {data.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-20 text-center text-slate-500">
                      {error ? "Connection failed." : "Click 'Start Simulation' to begin streaming..."}
                    </td>
                  </tr>
                ) : (
                  data.map((p) => (
                  <tr key={p.id} className="hover:bg-white/5 transition-colors group animate-in fade-in slide-in-from-top-1 duration-300">
                      <td className="py-4 px-6 font-mono text-slate-400">{p.time}</td>
                      <td className="py-4 px-6 font-mono font-bold text-white text-lg">{p.gru_rul}</td>
                      <td className="py-4 px-6 font-mono text-slate-300">{p.cnn_rul}</td>
                      <td className="py-4 px-6 font-mono text-slate-300">{p.lstm_rul}</td>
                      <td className="py-4 px-6 text-right">
                          <span className={`inline-flex items-center px-3 py-1 rounded-full text-[10px] font-bold tracking-wide border ${
                              p.status === 'NORMAL' ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400' : 
                              p.status === 'CRITICAL' ? 'border-rose-500/30 bg-rose-500/10 text-rose-400 shadow-glow' : 
                              'border-orange-500/30 bg-orange-500/10 text-orange-400'
                          }`}>
                              {p.status}
                          </span>
                      </td>
                  </tr>
                  ))
                )}
            </tbody>
            </table>
        </div>
      </div>
    </div>
  );
}