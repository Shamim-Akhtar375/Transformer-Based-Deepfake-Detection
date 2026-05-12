import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { UploadCloud, Image as ImageIcon, Video, Activity, CheckCircle, AlertTriangle, Loader2, ScanEye, Mic, Music, Waves } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api/v1';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('image'); // 'image' or 'video'
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const fileInputRef = useRef(null);

  const handleFileDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    processFile(droppedFile);
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    processFile(selectedFile);
  };

  const processFile = (file) => {
    if (!file) return;
    
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');
    const isAudio = file.type.startsWith('audio/') || file.type === 'video/webm'; // video/webm often used for audio blobs
    
    if (activeTab === 'image' && !isImage) {
      setError("Please upload an image file.");
      return;
    }
    if (activeTab === 'video' && !isVideo) {
      setError("Please upload a video file.");
      return;
    }
    if (activeTab === 'voice' && !isAudio) {
        setError("Please upload an audio file.");
        return;
    }
    
    setError(null);
    setFile(file);
    setResult(null);
    
    // Create preview
    const url = URL.createObjectURL(file);
    setPreview(url);
  };

  const startAnalysis = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      let endpoint = '/detect/image';
      if (activeTab === 'video') endpoint = '/detect/video';
      if (activeTab === 'voice') endpoint = '/detect/audio';
      
      const response = await axios.post(`${API_BASE}${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResult(response.data.prediction);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  const toggleRecording = async () => {
    if (isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Priority check for the best recording format supported by the browser
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
          ? 'audio/webm;codecs=opus' 
          : 'audio/webm';
          
        const options = { mimeType };
        const mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          audioChunksRef.current.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          const audioFile = new File([audioBlob], "recorded_audio.webm", { type: mimeType });
          processFile(audioFile);
          stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        setIsRecording(true);
      } catch (err) {
        setError("Microphone access denied or not available.");
      }
    }
  };

  const AudioVisualizer = ({ data }) => {
    if (!data) return null;
    const max = Math.max(...data.map(Math.abs));
    return (
      <div className="flex items-end gap-[1px] h-24 w-full bg-black/20 rounded-lg p-2 overflow-hidden">
        {data.map((val, i) => (
          <div 
            key={i} 
            className="flex-1 bg-primary/60 rounded-full" 
            style={{ height: `${Math.abs(val) / max * 100}%`, minHeight: '2px' }}
          />
        ))}
      </div>
    );
  };

  const VideoTimeline = ({ data }) => {
    if (!data || data.length === 0) return null;
    const height = 60;
    const width = 100;
    const points = data.map((d, i) => `${(i / (data.length - 1)) * width},${height - (d.fake_probability * height)}`).join(' ');

    return (
      <div className="w-full bg-black/20 rounded-xl p-4 border border-white/5">
        <div className="flex justify-between items-center mb-4">
            <h4 className="text-xs font-bold uppercase tracking-widest text-textMuted flex items-center gap-2">
                <Activity className="w-3 h-3 text-secondary" /> Temporal Anomaly Timeline
            </h4>
            <span className="text-[10px] font-mono text-secondary">PROBABILITY / TIME</span>
        </div>
        <div className="relative h-[80px] w-full">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full preserve-3d" preserveAspectRatio="none">
            <defs>
              <linearGradient id="grad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style={{ stopColor: 'theme(colors.secondary)', stopOpacity: 0.5 }} />
                <stop offset="100%" style={{ stopColor: 'theme(colors.secondary)', stopOpacity: 0 }} />
              </linearGradient>
            </defs>
            <path d={`M 0 ${height} L ${points} L ${width} ${height} Z`} fill="url(#grad)" />
            <polyline fill="none" stroke="theme(colors.secondary)" strokeWidth="0.5" points={points} className="drop-shadow-neon" />
            
            {/* Markers */}
            {data.map((d, i) => d.fake_probability > 0.5 && (
              <circle key={i} cx={(i / (data.length - 1)) * width} cy={height - (d.fake_probability * height)} r="1" fill="theme(colors.danger)" />
            ))}
          </svg>
          
          {/* Axis Labels */}
          <div className="absolute bottom-0 left-0 right-0 flex justify-between px-1 pt-1 border-t border-white/10 text-[8px] font-mono text-textMuted">
            <span>START</span>
            <span>{(data.length / 2).toFixed(0)}s</span>
            <span>END</span>
          </div>
        </div>
      </div>
    );
  };

  const FrameGallery = ({ frames }) => {
    if (!frames || frames.length === 0) return null;
    return (
      <div className="space-y-4">
        <h4 className="text-sm font-bold uppercase tracking-widest text-textMuted flex items-center gap-2">
            <ScanEye className="w-4 h-4 text-primary" /> Suspicious Frame Analysis
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {frames.map((frame, idx) => (
            <motion.div 
              key={idx}
              whileHover={{ scale: 1.05, y: -5 }}
              className="group relative rounded-xl overflow-hidden border border-white/10 bg-surfaceHighlight aspect-square cursor-pointer"
            >
              <img src={frame.image} alt={`Frame ${idx}`} className="w-full h-full object-cover grayscale group-hover:grayscale-0 transition-all duration-500" />
              
              {/* Heatmap Overlay on Hover */}
              {frame.heatmap && (
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                    <img src={`data:image/png;base64,${frame.heatmap}`} alt="Heatmap" className="w-full h-full object-cover mix-blend-screen" />
                </div>
              )}

              <div className="absolute top-2 left-2 px-2 py-0.5 bg-black/60 backdrop-blur-md rounded text-[10px] font-mono text-white border border-white/10">
                {frame.timestamp}
              </div>
              
              <div className={`absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent flex justify-between items-end`}>
                <div className="flex flex-col">
                    <span className="text-[10px] text-textMuted uppercase">Fake Prob</span>
                    <span className={`text-xs font-bold ${frame.confidence > 0.5 ? 'text-danger' : 'text-success'}`}>
                        {(frame.confidence * 100).toFixed(0)}%
                    </span>
                </div>
                <div className={`w-2 h-2 rounded-full ${frame.confidence > 0.5 ? 'bg-danger shadow-neon-danger' : 'bg-success shadow-neon'}`} />
              </div>
              
              {/* Manipulation Indicator */}
              {frame.confidence > 0.5 && (
                <div className="absolute inset-0 border-2 border-danger/50 rounded-xl pointer-events-none" />
              )}
            </motion.div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Activity className="text-primary" /> Forensic Analysis Scanner
          </h1>
          <p className="text-textMuted">Upload media for transformer-based deepfake detection.</p>
        </div>
        
        <div className="flex bg-surfaceHighlight rounded-lg p-1 border border-white/10">
          <button 
            onClick={() => { setActiveTab('image'); setFile(null); setPreview(null); setResult(null); }}
            className={`px-6 py-2 rounded-md font-medium transition-colors flex items-center gap-2 ${activeTab === 'image' ? 'bg-primary text-background' : 'text-textMuted hover:text-white'}`}
          >
            <ImageIcon className="w-4 h-4" /> Image
          </button>
          <button 
            onClick={() => { setActiveTab('video'); setFile(null); setPreview(null); setResult(null); }}
            className={`px-6 py-2 rounded-md font-medium transition-colors flex items-center gap-2 ${activeTab === 'video' ? 'bg-secondary text-white' : 'text-textMuted hover:text-white'}`}
          >
            <Video className="w-4 h-4" /> Video
          </button>
          <button 
            onClick={() => { setActiveTab('voice'); setFile(null); setPreview(null); setResult(null); }}
            className={`px-6 py-2 rounded-md font-medium transition-colors flex items-center gap-2 ${activeTab === 'voice' ? 'bg-accent text-white shadow-neon-accent' : 'text-textMuted hover:text-white'}`}
          >
            <Mic className="w-4 h-4" /> Voice
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload & Preview */}
        <div className="lg:col-span-2 space-y-6">
          <div 
            className={`glass-panel border-2 border-dashed ${file ? 'border-primary/50' : 'border-white/20'} rounded-2xl p-8 flex flex-col items-center justify-center text-center transition-colors min-h-[400px] relative overflow-hidden group`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => !file && fileInputRef.current?.click()}
          >
            {/* Background scanner animation */}
            {loading && (
              <motion.div 
                className="absolute left-0 right-0 h-1 bg-primary/80 shadow-neon z-10"
                animate={{ top: ["0%", "100%", "0%"] }}
                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              />
            )}

            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept={activeTab === 'image' ? "image/*" : activeTab === 'video' ? "video/*" : "audio/*"}
              onChange={handleFileSelect}
            />

            <AnimatePresence mode="wait">
              {!preview ? (
                <motion.div 
                  key="upload"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-4 cursor-pointer"
                >
                  <div className="p-6 bg-surfaceHighlight rounded-full group-hover:scale-110 transition-transform">
                    {activeTab === 'image' ? <ImageIcon className="w-12 h-12 text-primary" /> : 
                     activeTab === 'video' ? <Video className="w-12 h-12 text-secondary" /> : 
                     <Mic className="w-12 h-12 text-accent" />}
                  </div>
                  <div>
                    <p className="text-xl font-medium text-white mb-2">Drag & Drop {activeTab} here</p>
                    <p className="text-textMuted">or click to browse from your computer</p>
                  </div>
                  {activeTab === 'voice' && (
                    <button 
                      onClick={(e) => { e.stopPropagation(); toggleRecording(); }}
                      className={`mt-4 px-6 py-2 rounded-full flex items-center gap-2 transition-all ${isRecording ? 'bg-danger animate-pulse' : 'bg-accent/20 hover:bg-accent/40 border border-accent/50'}`}
                    >
                      {isRecording ? <><Activity className="w-4 h-4" /> Stop Recording</> : <><Mic className="w-4 h-4" /> Start Microphone</>}
                    </button>
                  )}
                </motion.div>
              ) : (
                <motion.div 
                  key="preview"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="w-full h-full flex flex-col items-center relative z-0"
                >
                  <div className="absolute top-4 right-4 z-20 flex gap-2">
                    <button 
                      onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null); }}
                      className="px-4 py-1 bg-black/50 hover:bg-danger/80 backdrop-blur-md rounded-md border border-white/10 text-white text-sm transition-colors"
                    >
                      Clear
                    </button>
                  </div>
                  
                  {activeTab === 'image' ? (
                    <img src={preview} alt="Preview" className="max-h-[350px] object-contain rounded-lg shadow-2xl" />
                  ) : activeTab === 'video' ? (
                    <div className="w-full space-y-4">
                        <video src={preview} controls className="max-h-[400px] rounded-lg w-full bg-black shadow-2xl border border-white/10" />
                        
                        {/* Interactive Timeline Markers */}
                        {result?.timeline && (
                            <div className="flex gap-1 h-3 w-full bg-surfaceHighlight rounded-full overflow-hidden p-[2px]">
                                {result.timeline.map((point, idx) => (
                                    <div 
                                        key={idx} 
                                        className={`flex-1 rounded-full transition-all ${point.fake_probability > 0.5 ? 'bg-danger animate-pulse' : 'bg-success/30'}`}
                                        title={`Timestamp: ${point.timestamp} | Prob: ${(point.fake_probability*100).toFixed(0)}%`}
                                    />
                                ))}
                            </div>
                        )}
                    </div>
                  ) : (
                    <div className="w-full max-w-md p-8 bg-surfaceHighlight rounded-2xl border border-white/10 flex flex-col items-center gap-6">
                        <div className="p-6 bg-accent/20 rounded-full">
                            <Music className="w-12 h-12 text-accent" />
                        </div>
                        <div className="w-full">
                            <audio src={preview} controls className="w-full" />
                        </div>
                        <p className="text-sm text-textMuted">{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</p>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {error && (
            <div className="bg-danger/10 border border-danger/50 text-danger px-4 py-3 rounded-lg flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              <p>{error}</p>
            </div>
          )}

          <div className="flex justify-end">
            <button 
              onClick={startAnalysis}
              disabled={!file || loading}
              className={`cyber-button text-lg w-full sm:w-auto ${(!file || loading) && 'opacity-50 cursor-not-allowed hover:bg-transparent hover:shadow-none before:hidden'}`}
            >
              {loading ? (
                <span className="flex items-center gap-2"><Loader2 className="animate-spin w-5 h-5" /> Executing Multimodal Pipelines...</span>
              ) : (
                <span className="flex items-center gap-2"><ScanEye className="w-5 h-5" /> Run Forensic Investigation</span>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Forensic Results */}
        <div className="lg:col-span-1">
          <div className="glass-panel p-6 h-full flex flex-col overflow-y-auto max-h-[800px] scrollbar-hide">
            <h2 className="text-xl font-bold border-b border-white/10 pb-4 mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-secondary" /> Forensic Report
            </h2>

            {!result && !loading && (
              <div className="flex-1 flex flex-col items-center justify-center text-textMuted text-center opacity-50">
                <ScanEye className="w-16 h-16 mb-4" />
                <p>Initialize scan to generate deepfake probability analysis.</p>
              </div>
            )}

            {loading && (
              <div className="flex-1 flex flex-col items-center justify-center space-y-4">
                <Loader2 className="w-12 h-12 text-primary animate-spin" />
                <div className="w-full space-y-2">
                  <div className="h-2 bg-surfaceHighlight rounded overflow-hidden">
                    <motion.div className="h-full bg-primary" initial={{ width: "0%" }} animate={{ width: "100%" }} transition={{ duration: 10, ease: "linear" }} />
                  </div>
                  <p className="text-xs text-center text-primary animate-pulse uppercase tracking-widest">Scanning frames for manipulation...</p>
                </div>
              </div>
            )}

            {result && (
              <motion.div 
                initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                {/* Verdict Summary Card */}
                <div className={`p-6 rounded-2xl border relative overflow-hidden flex flex-col items-center justify-center text-center ${result.label === 'FAKE' ? 'bg-danger/10 border-danger/50 shadow-neon-danger' : 'bg-success/10 border-success/50 shadow-neon'}`}>
                  {/* Glowing background effect */}
                  <div className={`absolute -top-10 -right-10 w-32 h-32 blur-[50px] opacity-20 ${result.label === 'FAKE' ? 'bg-danger' : 'bg-success'}`} />
                  
                  {result.label === 'FAKE' ? (
                    <AlertTriangle className="w-10 h-10 text-danger mb-2" />
                  ) : (
                    <CheckCircle className="w-10 h-10 text-success mb-2" />
                  )}
                  <h3 className={`text-4xl font-black tracking-tighter ${result.label === 'FAKE' ? 'text-danger' : 'text-success'}`}>
                    {result.label}
                  </h3>
                  <div className="flex items-center gap-4 mt-2">
                      <div className="flex flex-col">
                        <span className="text-[10px] text-textMuted uppercase">Confidence</span>
                        <span className="text-sm font-mono text-white">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-[1px] h-8 bg-white/10" />
                      <div className="flex flex-col">
                        <span className="text-[10px] text-textMuted uppercase">Severity</span>
                        <span className={`text-sm font-bold ${result.analysis_summary?.manipulation_severity === 'HIGH' ? 'text-danger' : 'text-success'}`}>
                            {result.analysis_summary?.manipulation_severity || 'LOW'}
                        </span>
                      </div>
                  </div>
                </div>

                {/* Video Specific: Timeline & Gallery */}
                {activeTab === 'video' && result.timeline && (
                    <div className="space-y-6">
                        <VideoTimeline data={result.timeline} />
                        
                        <div className="bg-surfaceHighlight p-4 rounded-xl border border-white/5 space-y-3">
                            <h4 className="text-xs font-bold uppercase tracking-widest text-textMuted flex items-center gap-2">
                                <Activity className="w-3 h-3 text-primary" /> Analysis Metrics
                            </h4>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-1">
                                    <span className="text-[10px] text-textMuted uppercase">Suspicious Frames</span>
                                    <p className="text-sm font-mono text-white">{result.analysis_summary?.suspicious_frame_count || 0}</p>
                                </div>
                                <div className="space-y-1">
                                    <span className="text-[10px] text-textMuted uppercase">Temporal Std</span>
                                    <p className="text-sm font-mono text-white">{(result.analysis_summary?.temporal_inconsistency || 0).toFixed(3)}</p>
                                </div>
                            </div>
                        </div>

                        <FrameGallery frames={result.suspicious_frames} />
                    </div>
                )}

                {/* Details / System Log */}
                <div className="bg-black/30 p-4 rounded-xl text-sm text-textMuted leading-relaxed border border-white/5 relative group">
                  <span className="text-primary font-mono text-[10px] uppercase mb-1 block flex justify-between">
                    <span>{">> "}Forensic Analysis</span>
                    <span className="text-textMuted opacity-50">{result.processing_time}</span>
                  </span>
                  {result.analysis}
                </div>

                {/* Heatmap for Image results (Original behavior) */}
                {activeTab === 'image' && result.heatmap && (
                  <div className="mt-4">
                    <h4 className="text-xs font-bold uppercase tracking-widest text-textMuted mb-2">Explainable AI: Saliency Map</h4>
                    <div className="rounded-xl overflow-hidden border border-white/10 bg-black shadow-2xl">
                      <img src={result.heatmap} alt="Grad-CAM" className="w-full mix-blend-screen" />
                    </div>
                  </div>
                )}

                {/* Audio Result Integration */}
                {result.audio_analyzed && (
                  <div className="bg-accent/10 border border-accent/20 p-4 rounded-xl flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-accent/20 rounded-lg">
                            <Mic className="w-4 h-4 text-accent" />
                        </div>
                        <div>
                            <p className="text-xs font-bold text-accent uppercase tracking-wider">Audio Forensic</p>
                            <p className="text-[10px] text-textMuted">Wav2Vec2 Analysis</p>
                        </div>
                    </div>
                    <span className={`text-xs font-bold px-3 py-1 rounded-full ${result.audio_prediction === 'FAKE' ? 'bg-danger/80 text-white shadow-neon-danger' : 'bg-success/80 text-black shadow-neon'}`}>
                      {result.audio_prediction}
                    </span>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
