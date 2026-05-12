import React from 'react';
import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Database, TrendingUp, Target } from 'lucide-react';

const accuracyData = [
  { epoch: '1', accuracy: 0.65, loss: 0.8 },
  { epoch: '5', accuracy: 0.75, loss: 0.6 },
  { epoch: '10', accuracy: 0.82, loss: 0.45 },
  { epoch: '20', accuracy: 0.89, loss: 0.3 },
  { epoch: '30', accuracy: 0.94, loss: 0.2 },
  { epoch: '40', accuracy: 0.96, loss: 0.15 },
  { epoch: '50', accuracy: 0.98, loss: 0.08 },
];

const modelComparison = [
  { name: 'EfficientNet-B4', f1: 0.89 },
  { name: 'Swin Transformer', f1: 0.92 },
  { name: 'ViT-Base', f1: 0.95 },
  { name: 'Multimodal Fusion', f1: 0.98 },
];

const Analytics = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
          <TrendingUp className="text-primary" /> Training Analytics & Metrics
        </h1>
        <p className="text-textMuted">Performance evaluation across various datasets and model architectures.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard title="Overall Accuracy" value="98.2%" icon={<Target className="text-success" />} />
        <StatCard title="Dataset Size (FaceForensics++)" value="1.8M Frames" icon={<Database className="text-secondary" />} />
        <StatCard title="Inference Time" value="45ms / frame" icon={<TrendingUp className="text-primary" />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6">
          <h3 className="text-xl font-bold mb-6 text-white">Training Convergence</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={accuracyData}>
                <defs>
                  <linearGradient id="colorAccuracy" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00F0FF" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00F0FF" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="epoch" stroke="#94A3B8" />
                <YAxis stroke="#94A3B8" />
                <Tooltip contentStyle={{ backgroundColor: '#0A1428', borderColor: '#00F0FF55' }} />
                <Area type="monotone" dataKey="accuracy" stroke="#00F0FF" fillOpacity={1} fill="url(#colorAccuracy)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-panel p-6">
          <h3 className="text-xl font-bold mb-6 text-white">Model Architecture Comparison (F1 Score)</h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
                <XAxis dataKey="name" stroke="#94A3B8" />
                <YAxis domain={[0.8, 1.0]} stroke="#94A3B8" />
                <Tooltip contentStyle={{ backgroundColor: '#0A1428', borderColor: '#6E00FF55' }} cursor={{ fill: '#ffffff05' }} />
                <Bar dataKey="f1" fill="#6E00FF" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon }) => (
  <div className="glass-panel p-6 flex items-center justify-between border-l-4 border-l-primary">
    <div>
      <p className="text-sm text-textMuted uppercase tracking-wider mb-1">{title}</p>
      <p className="text-3xl font-bold text-white font-mono">{value}</p>
    </div>
    <div className="p-4 bg-surfaceHighlight rounded-full">
      {icon}
    </div>
  </div>
);

export default Analytics;
