import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Shield, Zap, Cpu, ScanEye, FileText } from 'lucide-react';

const Landing = () => {
  return (
    <div className="relative min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center overflow-hidden">
      {/* Background Particles/Grid */}
      <div className="absolute inset-0 bg-cyber-grid pointer-events-none opacity-30" />
      
      <div className="max-w-6xl mx-auto px-4 z-10 text-center flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/30 bg-primary/10 text-primary text-sm font-medium mb-6">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-primary"></span>
            </span>
            System Online - v2.0 AI Core Active
          </div>
          <h1 className="text-6xl md:text-8xl font-black mb-6 tracking-tight">
            DEEP<span className="text-gradient">GUARD</span> PRO
          </h1>
          <p className="text-xl md:text-2xl text-textMuted max-w-3xl mx-auto font-light">
            Multimodal Transformer-Based Framework for Robust Deepfake Detection.
            Identify manipulated media with sub-pixel accuracy.
          </p>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-6 mb-20"
        >
          <Link to="/dashboard">
            <button className="cyber-button text-lg px-8 py-4 flex items-center gap-3">
              <ScanEye className="w-5 h-5" />
              Launch Scanner
            </button>
          </Link>
          <Link to="/research">
            <button className="px-8 py-4 bg-surfaceHighlight border border-white/10 hover:bg-white/5 rounded-none font-medium tracking-wide uppercase transition-colors flex items-center gap-3">
              <FileText className="w-5 h-5" />
              View Research
            </button>
          </Link>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full">
          <FeatureCard 
            icon={<Cpu className="w-8 h-8 text-primary" />}
            title="Vision Transformers"
            description="Leveraging state-of-the-art ViT architectures for spatial artifact detection in generated faces."
            delay={0.5}
          />
          <FeatureCard 
            icon={<Zap className="w-8 h-8 text-secondary" />}
            title="Temporal Fusion"
            description="Detecting inter-frame inconsistencies and lip-sync anomalies in synthetic videos."
            delay={0.6}
          />
          <FeatureCard 
            icon={<Shield className="w-8 h-8 text-success" />}
            title="Explainable AI"
            description="Generating Grad-CAM heatmaps to visualize manipulated regions for forensic analysis."
            delay={0.7}
          />
        </div>
      </div>
    </div>
  );
};

const FeatureCard = ({ icon, title, description, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.6 }}
    className="glass-panel p-8 text-left group hover:border-primary/50 transition-colors"
  >
    <div className="mb-4 p-3 bg-surfaceHighlight rounded-lg inline-block group-hover:scale-110 transition-transform">
      {icon}
    </div>
    <h3 className="text-xl font-bold mb-3">{title}</h3>
    <p className="text-textMuted leading-relaxed">{description}</p>
  </motion.div>
);

export default Landing;
