import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutGrid, Radio, Activity, GitGraph, Settings, Zap } from 'lucide-react';

const NavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `flex items-center gap-2 px-4 py-2 rounded-full transition-all duration-300 text-sm font-medium ${
        isActive 
          ? 'bg-primary/20 text-primary shadow-glow ring-1 ring-primary/50' 
          : 'text-slate-400 hover:text-white hover:bg-white/5'
      }`
    }
  >
    <Icon size={16} />
    <span>{label}</span>
  </NavLink>
);

export default function Layout() {
  return (
    <div className="min-h-screen bg-background text-text font-sans selection:bg-primary/30 flex flex-col">
      
      {/* TOP NAVIGATION BAR */}
      <nav className="sticky top-0 z-50 w-full bg-sidebar/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-tr from-primary to-secondary rounded-lg flex items-center justify-center text-white font-bold shadow-lg">
              M
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white hidden md:block">
              Metro<span className="text-primary">Pulse</span>
            </h1>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-2">
            <NavItem to="/" icon={LayoutGrid} label="Overview" />
            <NavItem to="/live" icon={Radio} label="Live Stream" />
            <NavItem to="/drift" icon={Zap} label="Drift" />
            <NavItem to="/compare" icon={GitGraph} label="Benchmarks" />
            <NavItem to="/retrain" icon={Settings} label="Pipeline" />
          </div>

          {/* Status Indicator */}
          <div className="hidden md:flex items-center gap-3 pl-6 border-l border-white/10">
            <div className="flex flex-col text-right">
                <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">System Status</span>
                <span className="text-xs font-mono text-emerald-400 flex items-center justify-end gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/> ONLINE
                </span>
            </div>
          </div>
        </div>
      </nav>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 relative w-full max-w-7xl mx-auto p-6 md:p-8">
        {/* Background Ambient Glow */}
        <div className="fixed top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none z-0" />
        
        {/* Content */}
        <div className="relative z-10">
          <Outlet />
        </div>
      </main>
    </div>
  );
}