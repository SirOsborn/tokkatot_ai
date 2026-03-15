import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Activity, 
  Database, 
  Cpu, 
  ShieldCheck, 
  Cloud, 
  BarChart3, 
  AlertTriangle, 
  Search, 
  CheckCircle2, 
  Github,
  Facebook,
  Globe,
  Mail,
  Microscope,
  Zap,
  Layers,
  Target,
  ArrowRightLeft,
  Server
} from 'lucide-react';

// --- Types ---

interface SlideProps {
  isActive: boolean;
  scale: number;
}

// --- Components ---

const Slide1: React.FC<SlideProps> = () => (
  <div className="flex flex-col justify-start pt-6 h-full relative overflow-hidden">
    <div className="absolute top-6 left-6 p-3 rounded-2xl bg-white/50 backdrop-blur-sm border border-zinc-100 shadow-sm z-20">
      <Microscope size={28} className="text-primary" />
    </div>
    <div className="absolute inset-0 -z-20 bg-gradient-to-br from-zinc-50 via-white to-primary/5" />
    <div className="absolute top-0 right-0 -z-10 opacity-30">
      <div className="w-[37.5rem] h-[37.5rem] bg-primary/20 rounded-full blur-[6.25rem] animate-pulse" />
    </div>
    <div className="absolute -bottom-24 -left-24 -z-10 opacity-20">
      <div className="w-[25rem] h-[25rem] bg-secondary/20 rounded-full blur-[5rem]" />
    </div>
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="space-y-6 relative z-10 pt-12"
    >
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary font-semibold text-sm uppercase tracking-wider">
        <ShieldCheck size={16} />
        Safety-First Deep Learning
      </div>
      <h1 className="text-4xl md:text-6xl font-bold leading-tight">
        Safety-First <span className="text-primary">Ensemble</span> Deep Learning
      </h1>
      <p className="text-xl md:text-2xl text-zinc-600 font-medium max-w-3xl">
        Revolutionizing Poultry Health Monitoring via Fecal Image Analysis
      </p>
      <div className="h-1 w-24 bg-secondary rounded-full" />
      <div className="pt-8 space-y-4">
        <div className="flex flex-wrap gap-x-12 gap-y-4">
          <div>
            <p className="text-sm text-zinc-400 uppercase tracking-widest">Presenter</p>
            <p className="text-lg font-semibold">Sun Heng</p>
          </div>
          <div>
            <p className="text-sm text-zinc-400 uppercase tracking-widest">Major</p>
            <p className="text-lg font-semibold">Data Science & AI Engineering</p>
          </div>
          <div>
            <p className="text-sm text-zinc-400 uppercase tracking-widest">Year / Term</p>
            <p className="text-lg font-semibold">Year 3 Term 1</p>
          </div>
          <div>
            <p className="text-sm text-zinc-400 uppercase tracking-widest">Course</p>
            <p className="text-lg font-semibold">Deep Learning</p>
          </div>
        </div>
      </div>
    </motion.div>
  </div>
);

const Slide2: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">The Background</h2>
    
    <div className="grid md:grid-cols-2 gap-8">
      <div className="space-y-4">
        <div className="space-y-2">
          <h3 className="text-xl font-bold text-supplementary flex items-center gap-3">
            <div className="p-2 rounded-lg bg-supplementary/10">
              <AlertTriangle size={20} />
            </div>
            The Crisis in Poultry Farming
          </h3>
          <div className="space-y-3">
            <div className="p-4 rounded-2xl bg-white shadow-sm border border-zinc-100 relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1 h-full bg-supplementary" />
              <p className="font-bold text-base mb-1">The Velocity of Disease</p>
              <p className="text-zinc-600 leading-relaxed text-sm">
                Outbreaks like <span className="text-supplementary font-semibold">Coccidiosis</span> or <span className="text-supplementary font-semibold">Newcastle Disease</span> can wipe out an entire flock in days.
              </p>
            </div>
            <div className="p-4 rounded-2xl bg-white shadow-sm border border-zinc-100 relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1 h-full bg-supplementary" />
              <p className="font-bold text-base mb-1">The Scalability Gap</p>
              <p className="text-zinc-600 leading-relaxed text-sm">
                Manual inspection of thousands of birds is labor-intensive and prone to human error.
              </p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="space-y-6">
        <div className="space-y-2">
          <h3 className="text-xl font-bold text-primary flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Target size={20} />
            </div>
            Tokkatot Vision
          </h3>
          <p className="text-zinc-600 text-base leading-relaxed">
            Transitioning from reactive veterinary care to <span className="text-primary font-bold">proactive, automated IoT health monitoring</span>.
          </p>
        </div>
        
        <div className="glass-card p-6 bg-primary/5 border-primary/20 relative overflow-hidden">
          <div className="absolute -right-4 -top-4 opacity-10">
            <ShieldCheck size={80} className="text-primary" />
          </div>
          <h4 className="font-bold text-primary mb-2 uppercase tracking-widest text-xs">The Deep Learning Goal</h4>
          <p className="text-xl font-serif italic text-zinc-800 leading-snug">
            "Move beyond 'General Accuracy'. Safety-First Paradigm: <span className="text-primary font-bold">Maximizing Recall</span> to ensure no sick bird is ever missed."
          </p>
        </div>
      </div>
    </div>
  </div>
);

const Slide3: React.FC<SlideProps> = () => (
  <div className="space-y-8">
    <h2 className="text-3xl font-bold accent-border">The Dataset: Foundation for Precision</h2>
    
    <div className="grid md:grid-cols-12 gap-6">
      <div className="md:col-span-4 space-y-4">
        <div className="glass-card p-6 flex flex-col items-center text-center space-y-3 border-b-8 border-primary h-full justify-center">
          <div className="w-16 h-16 rounded-3xl bg-primary/10 flex items-center justify-center text-primary shadow-inner">
            <Database size={32} />
          </div>
          <div>
            <p className="text-4xl font-black text-primary tracking-tighter">400,000+</p>
            <p className="text-xs text-zinc-400 font-bold uppercase tracking-widest mt-1">High-Res Images</p>
          </div>
          <p className="text-zinc-500 text-xs">Curated by avian pathology experts.</p>
        </div>
      </div>
      
      <div className="md:col-span-8 space-y-6">
        <div className="glass-card p-6">
          <h3 className="text-lg font-bold mb-4 flex items-center gap-3">
            <Layers size={20} className="text-secondary" />
            Four Target Classes
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {[
              { name: 'Coccidiosis', type: 'Parasitic', color: 'bg-supplementary' },
              { name: 'Healthy', type: 'Control', color: 'bg-primary' },
              { name: 'New Castle', type: 'Viral', color: 'bg-supplementary' },
              { name: 'Salmonella', type: 'Bacterial', color: 'bg-supplementary' }
            ].map((cls, i) => (
              <div key={i} className="flex items-center gap-3 p-3 rounded-2xl bg-zinc-50 border border-zinc-100 hover:border-zinc-300 transition-colors group">
                <div className={`w-3 h-3 rounded-full ${cls.color} shadow-sm group-hover:scale-125 transition-transform`} />
                <div>
                  <p className="font-bold text-base">{cls.name}</p>
                  <p className="text-[0.625rem] text-zinc-400 font-bold uppercase">{cls.type}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-2 space-y-3">
            <h3 className="text-lg font-bold flex items-center gap-2">
              <Zap size={18} className="text-secondary" />
              Preprocessing & Augmentation
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {[
                { icon: <ArrowRightLeft size={16} />, title: 'Geometric', desc: 'Rotations, Flips, Zooms' },
                { icon: <Zap size={16} />, title: 'Photometric', desc: 'Brightness, Contrast, Hue' },
                { icon: <Activity size={16} />, title: 'Normalization', desc: 'ImageNet Distribution' }
              ].map((item, i) => (
                <div key={i} className="p-3 rounded-2xl bg-white border border-zinc-100 shadow-sm">
                  <div className="text-primary mb-1">{item.icon}</div>
                  <p className="font-bold text-xs mb-0.5">{item.title}</p>
                  <p className="text-[0.5625rem] text-zinc-500 leading-tight">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="flex flex-col justify-end">
            <div className="p-4 rounded-3xl bg-zinc-900 text-white h-[6.5rem] flex flex-col justify-center">
              <h4 className="text-[0.625rem] font-bold text-zinc-500 uppercase tracking-widest mb-1">Data Source</h4>
              <p className="text-[0.6875rem] leading-tight">
                Multi-farm collection spanning diverse environmental conditions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Slide4: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">Model Selection (The "Backbones")</h2>
    
    <div className="grid md:grid-cols-2 gap-6">
      <div className="glass-card p-6 border-t-8 border-secondary relative overflow-hidden">
        <div className="absolute top-0 right-0 p-4 opacity-5">
          <Cpu size={80} />
        </div>
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-xl font-bold">EfficientNetB0</h3>
          <span className="px-2 py-0.5 rounded-full bg-secondary/10 text-secondary text-[0.5625rem] font-bold uppercase tracking-widest">Edge Specialist</span>
        </div>
        <ul className="space-y-2 mb-4 relative z-10">
          <li className="flex gap-2">
            <div className="p-0.5 rounded-full bg-secondary/10 text-secondary">
              <CheckCircle2 size={14} className="shrink-0" />
            </div>
            <p className="text-xs"><span className="font-bold">Efficiency:</span> Optimal scaling for mobile hardware.</p>
          </li>
          <li className="flex gap-2">
            <div className="p-0.5 rounded-full bg-secondary/10 text-secondary">
              <CheckCircle2 size={14} className="shrink-0" />
            </div>
            <p className="text-xs"><span className="font-bold">Role:</span> Real-time screening on Edge devices.</p>
          </li>
        </ul>
        <div className="flex gap-3">
          <div className="flex-1 p-2 rounded-xl bg-zinc-50 text-center">
            <p className="text-[0.5rem] text-zinc-400 font-bold uppercase">Params</p>
            <p className="text-base font-black text-secondary">5.3M</p>
          </div>
          <div className="flex-1 p-2 rounded-xl bg-zinc-50 text-center">
            <p className="text-[0.5rem] text-zinc-400 font-bold uppercase">FLOPs</p>
            <p className="text-base font-black text-secondary">0.39B</p>
          </div>
        </div>
      </div>
      
      <div className="glass-card p-6 border-t-8 border-primary relative overflow-hidden">
        <div className="absolute top-0 right-0 p-4 opacity-5">
          <Layers size={80} />
        </div>
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-xl font-bold">DenseNet121</h3>
          <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-[0.5625rem] font-bold uppercase tracking-widest">Feature Extractor</span>
        </div>
        <ul className="space-y-2 mb-4 relative z-10">
          <li className="flex gap-2">
            <div className="p-0.5 rounded-full bg-primary/10 text-primary">
              <CheckCircle2 size={14} className="shrink-0" />
            </div>
            <p className="text-xs"><span className="font-bold">Robustness:</span> Maximum feature reuse.</p>
          </li>
          <li className="flex gap-2">
            <div className="p-0.5 rounded-full bg-primary/10 text-primary">
              <CheckCircle2 size={14} className="shrink-0" />
            </div>
            <p className="text-xs"><span className="font-bold">Role:</span> High-precision Cloud verification.</p>
          </li>
        </ul>
        <div className="flex gap-3">
          <div className="flex-1 p-2 rounded-xl bg-zinc-50 text-center">
            <p className="text-[0.5rem] text-zinc-400 font-bold uppercase">Params</p>
            <p className="text-base font-black text-primary">8.0M</p>
          </div>
          <div className="flex-1 p-2 rounded-xl bg-zinc-50 text-center">
            <p className="text-[0.5rem] text-zinc-400 font-bold uppercase">Reuse</p>
            <p className="text-base font-black text-primary">High</p>
          </div>
        </div>
      </div>
    </div>
    
    <div className="glass-card p-6 bg-zinc-900 text-white relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent pointer-events-none" />
      <h3 className="text-lg font-bold mb-4 flex items-center gap-3">
        <Cpu size={20} className="text-primary" />
        Implementation Details
      </h3>
      <div className="grid md:grid-cols-3 gap-6">
        <div className="space-y-1">
          <p className="text-primary font-bold text-xs uppercase tracking-wider">Transfer Learning</p>
          <p className="text-zinc-400 text-[0.625rem] leading-relaxed">Initialized with ImageNet-1K weights.</p>
        </div>
        <div className="space-y-1">
          <p className="text-primary font-bold text-xs uppercase tracking-wider">Custom Heads</p>
          <p className="text-zinc-400 text-[0.625rem] leading-relaxed">2-layer FC head with GAP.</p>
        </div>
        <div className="space-y-1">
          <p className="text-primary font-bold text-xs uppercase tracking-wider">Regularization</p>
          <p className="text-zinc-400 text-[0.625rem] leading-relaxed">Dropout (0.3) and L2 Decay.</p>
        </div>
      </div>
    </div>
  </div>
);

const Slide5: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">Training Strategy (Recall-Focused Loss)</h2>
    
    <div className="grid md:grid-cols-2 gap-8 items-center">
      <div className="space-y-6">
        <div className="space-y-3">
          <h3 className="text-xl font-bold flex items-center gap-3">
            <div className="p-2 rounded-lg bg-supplementary/10 text-supplementary">
              <ArrowRightLeft size={20} />
            </div>
            Breaking the Symmetry of Error
          </h3>
          <p className="text-zinc-600 leading-relaxed text-sm">
            Standard Cross-Entropy treats False Negatives the same as False Positives. In poultry farming, a False Negative can lead to total flock loss.
          </p>
        </div>
        
        <div className="glass-card p-6 bg-supplementary/5 border-supplementary/20 relative overflow-hidden">
          <div className="absolute -right-4 -bottom-4 opacity-5">
            <Target size={80} className="text-supplementary" />
          </div>
          <h4 className="text-[0.625rem] font-bold text-supplementary mb-2 uppercase tracking-widest">RecallFocusedLoss (Custom)</h4>
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 rounded-2xl bg-white border border-supplementary/20 shadow-sm">
              <p className="text-2xl font-black text-supplementary">λ = 5</p>
            </div>
            <p className="text-xs font-bold text-zinc-400">Penalty Multiplier for False Negatives</p>
          </div>
          <p className="text-zinc-800 leading-relaxed text-sm">
            If the model misses a "Sick" bird, the loss is <span className="font-bold text-supplementary">5x higher</span>, forcing sensitivity.
          </p>
        </div>
      </div>
      
      <div className="space-y-4">
        <h3 className="text-xl font-bold flex items-center gap-3">
          <Zap size={20} className="text-primary" />
          Optimization Pipeline
        </h3>
        <div className="space-y-3">
          {[
            { step: 1, title: 'Optimizer: AdamW', desc: 'Superior regularization.', color: 'primary' },
            { step: 2, title: 'Scheduler: ReduceLROnPlateau', desc: 'Halves LR when recall stalls.', color: 'secondary' },
            { step: 3, title: 'Metric: Early Stopping', desc: 'Monitored on Validation Recall.', color: 'supplementary' }
          ].map((item, i) => (
            <div key={i} className="flex items-center gap-4 p-4 rounded-2xl bg-white shadow-sm border border-zinc-100">
              <div className={`w-10 h-10 rounded-full bg-${item.color}/10 flex items-center justify-center text-${item.color} font-black text-lg`}>
                {item.step}
              </div>
              <div>
                <p className="font-bold text-base">{item.title}</p>
                <p className="text-xs text-zinc-500">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
);

const Slide6: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">The "Safety-First" Ensemble Logic</h2>
    
    <div className="grid md:grid-cols-2 gap-8 items-center">
      <div className="space-y-6">
        <div className="glass-card p-6 space-y-4 border-l-8 border-primary">
          <h3 className="text-xl font-bold flex items-center gap-3 text-primary">
            <ShieldCheck size={24} />
            The Multi-Stage Decision Gate
          </h3>
          <div className="space-y-4">
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center font-bold shrink-0 text-xs">1</div>
              <div>
                <p className="font-bold text-sm">Parallel Voting</p>
                <p className="text-xs text-zinc-600">Independent verification by both models.</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center font-bold shrink-0 text-xs">2</div>
              <div>
                <p className="font-bold text-sm">Consensus Agreement</p>
                <p className="text-xs text-zinc-600">Only finalized if both models agree.</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center font-bold shrink-0 text-xs">3</div>
              <div>
                <p className="font-bold text-sm">Confidence Threshold</p>
                <p className="text-xs text-zinc-600">Softmax probability <span className="text-primary font-bold">&gt;85%</span>.</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="p-4 rounded-2xl bg-supplementary/5 border border-supplementary/20">
          <p className="text-zinc-800 font-medium italic text-center text-xs">
            "If models disagree OR confidence is low, the sample is routed for manual review. We prioritize 'I don't know' over 'I might be wrong'."
          </p>
        </div>
      </div>
      
      <div className="flex flex-col items-center gap-4">
        <img 
          src="assets/architecture_overview.png" 
          alt="Poultry Health System Overview" 
          className="rounded-2xl shadow-2xl border border-zinc-200 max-h-[18.75rem] object-contain bg-white p-3"
          referrerPolicy="no-referrer"
        />
        <p className="text-[0.625rem] text-zinc-400 font-bold uppercase tracking-widest">Fig 1: Tokkatot System Overview Architecture</p>
      </div>
    </div>
  </div>
);

const Slide7: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">System Architecture (Hybrid Edge-Cloud)</h2>
    
    <div className="grid md:grid-cols-12 gap-6">
      <div className="md:col-span-5 space-y-4">
        <div className="p-6 rounded-3xl bg-zinc-900 text-white relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-secondary/10 to-transparent opacity-50" />
          <h3 className="text-lg font-bold text-secondary mb-3 flex items-center gap-3">
            <Cpu size={20} />
            Edge Layer (Raspberry Pi)
          </h3>
          <ul className="space-y-3 text-zinc-400 relative z-10 text-xs">
            <li className="flex gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-secondary mt-1.5 shrink-0" />
              <p><span className="text-white font-bold">YOLOv8 Gatekeeper:</span> 24/7 real-time monitoring.</p>
            </li>
            <li className="flex gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-secondary mt-1.5 shrink-0" />
              <p><span className="text-white font-bold">EfficientNetB0:</span> Sub-100ms screening.</p>
            </li>
          </ul>
        </div>
        
        <div className="p-6 rounded-3xl bg-primary text-white relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent opacity-50" />
          <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-3">
            <Cloud size={20} />
            Cloud Layer (Tokkatot Server)
          </h3>
          <ul className="space-y-3 text-primary-foreground/80 relative z-10 text-xs">
            <li className="flex gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-white mt-1.5 shrink-0" />
              <p><span className="text-white font-bold">Full Ensemble:</span> High-precision verification.</p>
            </li>
            <li className="flex gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-white mt-1.5 shrink-0" />
              <p><span className="text-white font-bold">Vet Review:</span> Escalates uncertain cases.</p>
            </li>
          </ul>
        </div>
      </div>
      
      <div className="md:col-span-7 flex flex-col items-center justify-center gap-4">
        <img 
          src="assets/system_architecture_hierarchical.png" 
          alt="Hierarchical Architecture Overview" 
          className="rounded-2xl shadow-2xl border border-zinc-200 w-full h-64 object-contain bg-white p-4"
          referrerPolicy="no-referrer"
        />
        <p className="text-[0.625rem] text-zinc-400 font-bold uppercase tracking-widest">Fig 2: Tokkatot AI Hierarchical Architecture</p>
      </div>
    </div>
  </div>
);
const Slide8: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">Results: Performance Metrics</h2>
    
    <div className="grid md:grid-cols-12 gap-8 items-center">
      <div className="md:col-span-3 space-y-4">
        <div className="glass-card p-4 bg-primary/5 border-primary/20 relative overflow-hidden">
          <div className="absolute -right-8 -bottom-8 opacity-5">
            <BarChart3 size={80} />
          </div>
          <h3 className="text-base font-bold text-primary mb-1">Benchmark Success</h3>
          <div className="flex items-baseline gap-2">
            <p className="text-3xl font-black text-primary">99.1%</p>
            <p className="text-zinc-500 font-bold uppercase tracking-widest text-[0.5rem]">Accuracy</p>
          </div>
          <p className="text-zinc-600 mt-1 text-[0.625rem] leading-tight">
            Achieved on all samples cleared by the ensemble voting gate.
          </p>
        </div>
        
        <div className="p-3 rounded-2xl bg-zinc-100 border border-zinc-200">
          <h4 className="font-bold text-sm mb-1 flex items-center gap-2">
            <Zap size={14} className="text-secondary" />
            Ensemble Advantage
          </h4>
          <p className="text-zinc-600 leading-relaxed text-[0.625rem]">
            The ensemble reduced the error rate by <span className="font-bold text-primary text-sm">40%</span> compared to using EfficientNetB0 alone.
          </p>
        </div>
      </div>
      
      <div className="md:col-span-9 flex flex-col items-center gap-4">
        <div className="w-full">
          <img 
            src="assets/model_performance_comparison.png" 
            alt="Overall Model Performance Comparison" 
            className="rounded-2xl shadow-2xl border border-zinc-200 w-full h-[26rem] object-contain bg-white p-6"
            referrerPolicy="no-referrer"
          />
        </div>
        <div className="grid grid-cols-3 gap-3 w-full">
          <div className="p-2 rounded-xl bg-white border border-zinc-100 text-center">
            <p className="text-[0.4375rem] text-zinc-400 font-bold uppercase">Precision</p>
            <p className="text-xs font-black text-primary">0.992</p>
          </div>
          <div className="p-2 rounded-xl bg-white border border-zinc-100 text-center">
            <p className="text-[0.4375rem] text-zinc-400 font-bold uppercase">Recall</p>
            <p className="text-xs font-black text-secondary">0.998</p>
          </div>
          <div className="p-2 rounded-xl bg-white border border-zinc-100 text-center">
            <p className="text-[0.4375rem] text-zinc-400 font-bold uppercase">F1-Score</p>
            <p className="text-xs font-black text-supplementary">0.995</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Slide9: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">Results: Safety Statistics</h2>
    
    <div className="grid md:grid-cols-12 gap-6">
      <div className="md:col-span-4 space-y-4">
        <div className="glass-card p-6 space-y-3 border-t-8 border-primary">
          <h3 className="text-xl font-black text-primary">The "5.01% Rule"</h3>
          <div className="space-y-4">
            <div className="p-4 rounded-2xl bg-primary/5 border border-primary/10">
              <p className="text-[0.625rem] text-primary font-bold uppercase tracking-widest mb-1">Automation Rate</p>
              <p className="text-3xl font-black">94.99%</p>
              <p className="text-[0.625rem] text-zinc-500 mt-1">Classified with near-zero error.</p>
            </div>
            <div className="p-4 rounded-2xl bg-supplementary/5 border border-supplementary/10">
              <p className="text-[0.625rem] text-supplementary font-bold uppercase tracking-widest mb-1">Isolation Rate</p>
              <p className="text-3xl font-black">5.01%</p>
              <p className="text-[0.625rem] text-zinc-500 mt-1">Triggered manual vet review.</p>
            </div>
          </div>
        </div>
        
        <div className="p-6 rounded-3xl bg-zinc-900 text-white shadow-xl">
          <h3 className="text-base font-bold text-secondary mb-2 uppercase tracking-widest">Business Value</h3>
          <p className="text-zinc-400 leading-relaxed text-xs">
            By isolating 5% of ambiguous cases, we achieve <span className="text-white font-bold">Total Flock Safety</span>.
          </p>
        </div>
      </div>
      
      <div className="md:col-span-8 flex flex-col items-center justify-center gap-4">
        <img 
          src="assets/isolation_statistics.png" 
          alt="Safety Routing Statistics" 
          className="rounded-2xl shadow-2xl border border-zinc-200 w-full h-64 object-contain bg-white p-4"
          referrerPolicy="no-referrer"
        />
        <p className="text-[0.625rem] text-zinc-400 font-bold uppercase tracking-widest">Fig 3: Ensemble Safety Routing & Distribution</p>
      </div>
    </div>
  </div>
);

const Slide10: React.FC<SlideProps> = () => (
  <div className="space-y-6">
    <h2 className="text-3xl font-bold accent-border">Error Analysis & Discussion</h2>
    
    <div className="grid md:grid-cols-12 gap-6">
      <div className="md:col-span-7 space-y-4">
        <div className="glass-card p-6 h-full flex flex-col justify-center">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2 text-secondary">
            <BarChart3 size={24} />
            Performance Heatmap Analysis
          </h3>
          <img 
            src="assets/metrics_heatmap.png" 
            alt="Performance Heatmap" 
            className="rounded-2xl border border-zinc-100 w-full object-contain h-80 bg-white p-4 shadow-lg" 
            referrerPolicy="no-referrer" 
          />
          <p className="text-[0.625rem] text-zinc-400 font-bold uppercase tracking-widest mt-4 text-center">Fig 4: Per-Class Performance Metrics Heatmap</p>
        </div>
      </div>
      
      <div className="md:col-span-5 space-y-4">
        <div className="p-6 rounded-3xl bg-white shadow-sm border border-zinc-100 space-y-4">
          <h3 className="text-xl font-bold">Understanding the "Uncertain"</h3>
          <div className="space-y-3">
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-supplementary/10 flex items-center justify-center text-supplementary shrink-0">
                <AlertTriangle size={16} />
              </div>
              <div>
                <p className="font-bold text-sm">Visual Ambiguity</p>
                <p className="text-xs text-zinc-500">Errors primarily occur between Coccidiosis and Salmonella.</p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center text-secondary shrink-0">
                <Search size={16} />
              </div>
              <div>
                <p className="font-bold text-sm">Environmental Noise</p>
                <p className="text-xs text-zinc-500">Bedding material interfering with texture detection.</p>
              </div>
            </div>
          </div>
          
          <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 text-center">
            <p className="text-[0.625rem] font-bold text-primary uppercase tracking-widest mb-1">Mitigation Success</p>
            <p className="text-2xl font-black text-primary">98%</p>
            <p className="text-[0.625rem] text-zinc-600 mt-1">of ambiguous cases caught.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Slide11: React.FC<SlideProps> = () => (
  <div className="space-y-8">
    <h2 className="text-3xl font-bold accent-border">Conclusion & Future Work</h2>
    
    <div className="max-w-4xl mx-auto space-y-10">
      <div className="space-y-6">
        <h3 className="text-2xl font-bold text-primary flex items-center gap-3">
          <CheckCircle2 size={28} />
          Key Takeaways
        </h3>
        <div className="grid grid-cols-1 gap-4">
          <div className="flex gap-4 p-6 rounded-3xl bg-white shadow-md border border-zinc-100 items-center">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
              <Zap size={24} />
            </div>
            <p className="text-zinc-800 font-semibold text-lg">Developed a dual-model ensemble that perfectly balances processing speed with diagnostic reliability.</p>
          </div>
          <div className="flex gap-4 p-6 rounded-3xl bg-white shadow-md border border-zinc-100 items-center">
            <div className="w-12 h-12 rounded-2xl bg-secondary/10 flex items-center justify-center text-secondary shrink-0">
              <ShieldCheck size={24} />
            </div>
            <p className="text-zinc-800 font-semibold text-lg">Shifted the Deep Learning paradigm to "Flock Safety" through custom loss functions and safety routing.</p>
          </div>
        </div>
      </div>
      
      <div className="space-y-6">
        <h3 className="text-2xl font-bold text-secondary flex items-center gap-3">
          <Activity size={28} />
          Next Steps for Tokkatot
        </h3>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="flex items-center gap-4 p-5 rounded-2xl bg-secondary/5 border border-secondary/20 shadow-sm">
            <Globe size={24} className="text-secondary" />
            <p className="text-base font-bold">Field Trials: Cambodia Pilot (Q1 2026)</p>
          </div>
          <div className="flex items-center gap-4 p-5 rounded-2xl bg-secondary/5 border border-secondary/20 shadow-sm">
            <Activity size={24} className="text-secondary" />
            <p className="text-base font-bold">Model Expansion: Ducks and Quails</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

const Slide12: React.FC<SlideProps> = () => (
  <div className="flex flex-col items-center justify-center h-full text-center space-y-12 relative overflow-hidden">
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -z-10 opacity-10">
      <div className="w-[50rem] h-[50rem] bg-primary/20 rounded-full blur-[7.5rem]" />
    </div>
    
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className="space-y-6"
    >
      <div className="inline-flex items-center gap-2 px-6 py-2 rounded-full bg-primary/10 text-primary font-black text-sm uppercase tracking-[0.2em]">
        End of Presentation
      </div>
      <h2 className="text-6xl md:text-7xl font-black text-zinc-900 tracking-tighter">
        Thank <span className="text-primary italic">You!</span>
      </h2>
      <p className="text-xl text-zinc-500 font-medium">Empowering Farmers with Safety-First AI</p>
    </motion.div>
    
    <div className="grid md:grid-cols-3 gap-6 w-full max-w-5xl">
      {[
        { icon: <Github size={32} />, label: 'Project Repo', value: 'github.com/SirOsborn/tokkatot_ai', link: 'https://github.com/SirOsborn/tokkatot_ai', color: 'text-zinc-900' },
        { icon: <Globe size={32} />, label: 'Startup Info', value: 'tokkatot.aztrolabe.com', link: 'https://tokkatot.aztrolabe.com', color: 'text-primary' },
        { icon: <Facebook size={32} />, label: 'Facebook', value: 'តុក្កតត - Tokkatot', link: 'https://facebook.com/tokkatot', color: 'text-blue-600' }
      ].map((item, i) => (
        <a key={i} href={item.link} target="_blank" className="glass-card p-8 hover:bg-white hover:shadow-xl hover:-translate-y-1 transition-all group">
          <div className={`mb-4 ${item.color} transition-colors flex justify-center`}>
            {item.icon}
          </div>
          <p className="text-[0.625rem] font-bold text-zinc-400 uppercase tracking-widest mb-1">{item.label}</p>
          <p className="text-sm font-bold truncate text-zinc-800">{item.value}</p>
        </a>
      ))}
    </div>
    
    <div className="pt-12 flex flex-col items-center gap-4">
      <p className="text-xs text-zinc-400 font-bold uppercase tracking-widest">Questions & Discussion</p>
    </div>
  </div>
);

// --- Main Presentation Component ---

const Presentation: React.FC = () => {
  const [currentSlide, setCurrentSlide] = React.useState(0);
  const [scale, setScale] = React.useState(1);
  const slides = [
    Slide1, Slide2, Slide3, Slide4, Slide5, Slide6, 
    Slide7, Slide8, Slide9, Slide10, Slide11, Slide12
  ];

  const nextSlide = () => setCurrentSlide((prev) => Math.min(prev + 1, slides.length - 1));
  const prevSlide = () => setCurrentSlide((prev) => Math.max(prev - 1, 0));

  const handleScreenClick = (e: React.MouseEvent) => {
    // Ignore clicks on links or interactive elements
    if ((e.target as HTMLElement).closest('a, button, input, [role="button"]')) return;
    
    const { clientX } = e;
    const { innerWidth } = window;
    if (clientX < innerWidth / 2) {
      prevSlide();
    } else {
      nextSlide();
    }
  };

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (['ArrowRight', 'ArrowDown', ' ', 'PageDown'].includes(e.key)) {
        e.preventDefault();
        nextSlide();
      }
      if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) {
        e.preventDefault();
        prevSlide();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const CurrentSlideComponent = slides[currentSlide];

  React.useEffect(() => {
    const updateFontSize = () => {
      // Base resolution for scaling (1100x700)
      // Using a slightly smaller base makes the content grow more on standard screens
      const baseWidth = 1100;
      const baseHeight = 700;
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      
      const scaleX = windowWidth / baseWidth;
      const scaleY = windowHeight / baseHeight;
      
      // Use the smaller scale to ensure content fits both dimensions
      const newScale = Math.min(scaleX, scaleY);
      setScale(newScale);
      
      // Set root font size (default is 16px)
      // We allow it to grow significantly for full screen
      const newFontSize = Math.max(12, 16 * newScale);
      document.documentElement.style.fontSize = `${newFontSize}px`;
    };

    window.addEventListener('resize', updateFontSize);
    updateFontSize();
    
    return () => {
      window.removeEventListener('resize', updateFontSize);
      document.documentElement.style.fontSize = ''; // Reset on unmount
    };
  }, []);

  return (
    <div className="slide-container cursor-pointer" onClick={handleScreenClick}>
      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full -z-20 overflow-hidden pointer-events-none">
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute top-1/2 -right-24 w-64 h-64 bg-secondary/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-24 left-1/2 w-80 h-80 bg-supplementary/5 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="p-6 flex justify-end items-center z-10">
        <div className="text-sm font-black text-primary tracking-widest">
          {currentSlide + 1}
        </div>
      </header>

      {/* Slide Content */}
      <main className="slide-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentSlide}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className="flex-1"
          >
            <CurrentSlideComponent isActive={true} scale={scale} />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer / Navigation */}
      <footer className="p-6 flex justify-between items-center z-10">
        <div className="text-xs font-medium text-zinc-400">
          Tokkatot | Agri-Tech Solutions | {new Date().getFullYear()}
        </div>
      </footer>

      {/* Progress Bar */}
      <div className="absolute bottom-0 left-0 h-1 bg-zinc-100 w-full">
        <motion.div 
          className="h-full bg-primary"
          initial={{ width: 0 }}
          animate={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  );
};

export default Presentation;
