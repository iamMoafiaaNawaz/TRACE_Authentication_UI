import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Sparkles, Layers, Activity, Eye, Database, ShieldCheck, CheckCircle2 } from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';

const About = () => {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-700">
      
      {/* 1. Navigation Bar (UPDATED) */}
      <nav className="bg-white/90 backdrop-blur-md sticky top-0 z-50 border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center">
          
          {/* A. Back Button (Ab Left Side par hai) */}
          <Link 
            to="/" 
            className="flex items-center gap-2 text-sm font-bold text-slate-500 hover:text-[#1E90FF] transition-colors mr-6"
          >
            <ArrowLeft size={20} /> Back to Home
          </Link>

          {/* B. Vertical Divider (Design ke liye) */}
          <div className="h-8 w-px bg-slate-200 mr-6 hidden sm:block"></div>

          {/* C. Logo Section (Right of button, NOT Clickable) */}
          <div className="flex items-center gap-3 select-none"> {/* select-none prevents highlighting */}
            <img src={traceLogo} alt="TRACE" className="h-10 w-auto object-contain" />
            <div className="flex flex-col">
              <span className="text-lg font-bold text-slate-800 leading-none">TRACE</span>
              <span className="text-[10px] font-semibold text-[#1E90FF] uppercase tracking-wide">About System</span>
            </div>
          </div>

        </div>
      </nav>

      {/* 2. Hero Section: Definition */}
      <section className="bg-white py-16 px-6 border-b border-slate-100">
        <div className="max-w-5xl mx-auto text-center">
          <span className="inline-block py-1 px-3 rounded-full bg-blue-50 text-[#1E90FF] text-xs font-bold tracking-widest uppercase mb-4 border border-blue-100">
            Clinical Decision Support System
          </span>
          <h1 className="text-4xl md:text-6xl font-extrabold text-slate-900 mb-2 tracking-tight">
            TRACE
          </h1>
          
          {/* UPDATED FULL FORM */}
          <h2 className="text-lg md:text-2xl font-medium text-[#1E90FF] mb-6 max-w-3xl mx-auto leading-relaxed">
            Transparent Recognition with Artifact Cleanup and Explainability
          </h2>
          
          <p className="text-lg text-slate-600 leading-relaxed max-w-3xl mx-auto">
            A web-based platform designed to assist dermatologists and medical trainees in the early detection of skin cancer. 
            We focus on delivering <span className="text-slate-900 font-semibold">reliable predictions</span>, 
            <span className="text-slate-900 font-semibold"> artifact-free images</span>, and 
            <span className="text-slate-900 font-semibold"> complete interpretability</span> across every step of the diagnostic workflow.
          </p>
        </div>
      </section>

      {/* 3. The Core Capabilities (Grid) */}
      <section className="py-16 px-6 bg-slate-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">What TRACE Does</h2>
            <p className="text-slate-500 max-w-2xl mx-auto">
              Addressing the gaps in modern dermatoscopic analysis through a fully engineered pipeline.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            
            {/* Feature 1: Preprocessing */}
            <FeatureCard 
              icon={<Sparkles className="text-yellow-500" />}
              title="1. Intelligent Preprocessing"
              desc="Standardizing images to ensure models receive clinically clean inputs."
              points={[
                "Hair artifact removal",
                "Illumination normalization",
                "Contrast enhancement",
                "Color & Texture alignment"
              ]}
            />

            {/* Feature 2: Fusion */}
            <FeatureCard 
              icon={<Layers className="text-blue-500" />}
              title="2. Texture-Deep Fusion"
              desc="Combining Local Binary Patterns (LBP) with CNN models to capture micro-textures."
              points={[
                "Captures pigment networks",
                "Recognizes subtle indicators",
                "Robust against noise",
                "Reduces color-only dependency"
              ]}
            />

            {/* Feature 3: Classification */}
            <FeatureCard 
              icon={<Activity className="text-red-500" />}
              title="3. Multi-Class Classification"
              desc="Backend powered by fine-tuned CNNs (VGG16, ResNet, EfficientNet)."
              points={[
                "NV – Melanocytic Nevus",
                "MEL – Melanoma",
                "BKL – Benign Keratosis",
                "BCC – Basal Cell Carcinoma"
              ]}
            />

            {/* Feature 4: Explainability */}
            <FeatureCard 
              icon={<Eye className="text-purple-500" />}
              title="4. Explainable AI (XAI)"
              desc="Building clinical trust by rendering 'Why' a model made a prediction."
              points={[
                "Activation map visualization",
                "Grad-CAM / Grad-CAM++",
                "Attribution heatmaps",
                "Feature importance summaries"
              ]}
            />

            {/* Feature 5: Audit Trail */}
            <FeatureCard 
              icon={<Database className="text-green-500" />}
              title="5. Full Audit Trail"
              desc="Every stage is logged in MongoDB based on the TRACE JSON-tree structure."
              points={[
                "Raw image → Final diagnosis trace",
                "Experiment reproducibility",
                "Experiment versioning",
                "Clinically auditable paths"
              ]}
            />

            {/* Why it matters Card */}
            <div className="bg-[#1E90FF] p-8 rounded-2xl shadow-lg text-white flex flex-col justify-center">
              <ShieldCheck className="w-12 h-12 mb-4 text-white/80" />
              <h3 className="text-xl font-bold mb-3">Why TRACE Matters</h3>
              <p className="text-blue-100 text-sm leading-relaxed">
                We transform raw dermatoscopic images into clinically meaningful insights while documenting each step for medical accountability.
              </p>
              <div className="mt-6 pt-6 border-t border-white/20">
                <p className="text-xs font-semibold uppercase tracking-widest text-white/70">Designed For</p>
                <p className="text-sm font-medium mt-1">Dermatologists • Students • Researchers</p>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* 4. Footer Disclaimer */}
      <footer className="bg-white border-t border-slate-200 py-10 px-6 text-center">
        <div className="max-w-3xl mx-auto">
          <p className="text-sm mb-2 font-bold text-slate-400 uppercase tracking-wide">Medical Disclaimer</p>
          <p className="text-xs text-slate-500 leading-relaxed mb-6">
            TRACE is designed for educational and decision-support purposes only. 
            It is not intended to replace professional medical diagnosis. 
            Final clinical decisions should always be made by a qualified healthcare provider.
          </p>
          <p className="text-xs text-slate-400 font-medium">© 2025 TRACE System. All rights reserved.</p>
        </div>
      </footer>

    </div>
  );
};

// Reusable Feature Card Component
const FeatureCard = ({ icon, title, desc, points }) => (
  <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md hover:border-blue-100 transition-all duration-300">
    <div className="flex items-center gap-4 mb-4">
      <div className="w-12 h-12 bg-slate-50 rounded-xl flex items-center justify-center shrink-0">
        {icon}
      </div>
      <h3 className="text-lg font-bold text-slate-800 leading-tight">{title}</h3>
    </div>
    
    <p className="text-slate-600 text-sm mb-6 leading-relaxed">
      {desc}
    </p>

    <ul className="space-y-2">
      {points.map((point, index) => (
        <li key={index} className="flex items-start gap-2 text-xs text-slate-500 font-medium">
          <CheckCircle2 className="w-4 h-4 text-[#1E90FF] shrink-0" />
          {point}
        </li>
      ))}
    </ul>
  </div>
);

export default About;