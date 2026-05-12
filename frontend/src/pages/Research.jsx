import React from 'react';
import { BookOpen, Award, CheckCircle } from 'lucide-react';

const Research = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 py-12 space-y-12">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-4 border border-primary/20">
          <Award className="w-4 h-4" /> Final Year Project / IEEE Research
        </div>
        <h1 className="text-4xl md:text-5xl font-black text-white">
          Multimodal Transformer-Based Framework for Robust Deepfake Detection
        </h1>
        <p className="text-xl text-textMuted max-w-3xl mx-auto leading-relaxed">
          A novel approach combining Vision Transformers (ViT) and temporal fusion 
          strategies to detect highly sophisticated synthetic media manipulations.
        </p>
      </div>

      <div className="glass-panel p-8 space-y-6 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <BookOpen className="text-primary" /> Abstract
        </h2>
        <p className="text-textMuted leading-relaxed text-lg">
          The proliferation of hyper-realistic deepfakes poses a significant threat to digital trust and security. 
          Traditional CNN-based detection methods often fail to generalize to unseen manipulation techniques and 
          struggle with heavy compressions. In this research, we propose a Multimodal Transformer-Based Framework 
          that leverages self-attention mechanisms to capture both spatial artifacts (e.g., blending anomalies, 
          frequency inconsistencies) and temporal mismatch. Evaluated on the FaceForensics++ C23 dataset, our 
          fusion model achieves an accuracy of 98.2%, outperforming state-of-the-art baselines.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="glass-panel p-8 space-y-4">
          <h2 className="text-xl font-bold text-white mb-4">Methodology</h2>
          <ul className="space-y-4 text-textMuted">
            <li className="flex gap-3">
              <CheckCircle className="text-success w-5 h-5 flex-shrink-0 mt-1" />
              <span><strong>Spatial Extraction:</strong> Utilizing ViT-Base models to analyze patch-level anomalies in facial regions.</span>
            </li>
            <li className="flex gap-3">
              <CheckCircle className="text-success w-5 h-5 flex-shrink-0 mt-1" />
              <span><strong>Temporal Modeling:</strong> Processing consecutive frames using TimeSformer to capture inconsistent physiological cues (e.g., eye blinking patterns).</span>
            </li>
            <li className="flex gap-3">
              <CheckCircle className="text-success w-5 h-5 flex-shrink-0 mt-1" />
              <span><strong>Explainability:</strong> Implementing Grad-CAM to visualize the exact regions the transformer attends to when predicting 'FAKE'.</span>
            </li>
          </ul>
        </div>
        
        <div className="glass-panel p-8 space-y-4">
          <h2 className="text-xl font-bold text-white mb-4">Key Contributions</h2>
          <ul className="space-y-4 text-textMuted">
            <li className="flex gap-3">
              <CheckCircle className="text-secondary w-5 h-5 flex-shrink-0 mt-1" />
              <span>Robustness against extreme video compression (H.264 / C23 level).</span>
            </li>
            <li className="flex gap-3">
              <CheckCircle className="text-secondary w-5 h-5 flex-shrink-0 mt-1" />
              <span>Cross-dataset generalization capabilities on unseen GAN variations.</span>
            </li>
            <li className="flex gap-3">
              <CheckCircle className="text-secondary w-5 h-5 flex-shrink-0 mt-1" />
              <span>Real-time inference optimized via ONNX Runtime for deployment.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Research;
