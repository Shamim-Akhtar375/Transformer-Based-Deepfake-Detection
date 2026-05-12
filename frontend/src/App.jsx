import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Research from './pages/Research';
import { Shield, LayoutDashboard, LineChart, FileText } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <nav className="border-b border-white/10 bg-surface/50 backdrop-blur-lg sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-2">
                <Shield className="w-8 h-8 text-primary" />
                <span className="font-bold text-xl tracking-tight">DeepGuard<span className="text-primary">Pro</span></span>
              </div>
              <div className="flex space-x-8">
                <Link to="/" className="text-textMuted hover:text-primary transition-colors font-medium flex items-center gap-2">
                   Home
                </Link>
                <Link to="/dashboard" className="text-textMuted hover:text-primary transition-colors font-medium flex items-center gap-2">
                  <LayoutDashboard className="w-4 h-4" /> Dashboard
                </Link>
                <Link to="/analytics" className="text-textMuted hover:text-primary transition-colors font-medium flex items-center gap-2">
                  <LineChart className="w-4 h-4" /> Analytics
                </Link>
                <Link to="/research" className="text-textMuted hover:text-primary transition-colors font-medium flex items-center gap-2">
                  <FileText className="w-4 h-4" /> Research
                </Link>
              </div>
            </div>
          </div>
        </nav>

        <main className="flex-1 w-full relative">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/research" element={<Research />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
