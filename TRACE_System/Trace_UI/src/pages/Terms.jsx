import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, ShieldAlert, Lock, Scale, FileText, CheckCircle2, AlertTriangle } from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';

const Terms = () => {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-700">
      
      {/* 1. Navigation Bar */}
      <nav className="bg-white/90 backdrop-blur-md sticky top-0 z-50 border-b border-slate-200">
        <div className="max-w-5xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img src={traceLogo} alt="TRACE" className="h-10 w-auto object-contain" />
            <div className="flex flex-col">
              <span className="text-lg font-bold text-slate-800 leading-none">TRACE</span>
              <span className="text-[10px] font-semibold text-[#1E90FF] uppercase tracking-wide">Terms of Use</span>
            </div>
          </div>
          <Link 
            to="/" 
            className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-[#1E90FF] transition-colors"
          >
            <ArrowLeft size={18} /> Back to Home
          </Link>
        </div>
      </nav>

      {/* 2. Header Section */}
      <section className="bg-white py-12 px-6 border-b border-slate-100">
        <div className="max-w-3xl mx-auto text-center">
          <div className="inline-flex items-center justify-center p-3 bg-blue-50 rounded-full mb-4">
            <Scale className="text-[#1E90FF] w-6 h-6" />
          </div>
          <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4">
            Terms of Service
          </h1>
          <p className="text-slate-500 text-sm">
            Last Updated: December 2025
          </p>
          <p className="text-lg text-slate-600 mt-6 leading-relaxed">
            Please read these terms carefully before using the TRACE (Transparent Reliable Automated Clinical Evaluation) system. 
            By accessing or using our platform, you agree to be bound by these terms.
          </p>
        </div>
      </section>

      {/* 3. Main Content Terms */}
      <div className="max-w-4xl mx-auto px-6 py-12 space-y-8">

        {/* Section 1: The Critical Disclaimer (Highlighted) */}
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
            <AlertTriangle size={100} className="text-amber-500" />
          </div>
          <div className="relative z-10">
            <h2 className="text-xl font-bold text-amber-900 mb-3 flex items-center gap-2">
              <ShieldAlert className="w-6 h-6" /> 1. Medical Disclaimer
            </h2>
            <p className="text-amber-800/80 leading-relaxed mb-4">
              <strong>TRACE is a Clinical Decision Support System (DSS) and an educational tool.</strong>
            </p>
            <ul className="space-y-2 text-sm text-amber-900 font-medium">
              <li className="flex gap-2">
                <CheckCircle2 size={16} className="shrink-0 mt-0.5" />
                <span>The AI analysis provided is NOT a medical diagnosis.</span>
              </li>
              <li className="flex gap-2">
                <CheckCircle2 size={16} className="shrink-0 mt-0.5" />
                <span>It is intended to serve as a "Second Opinion" to assist qualified professionals.</span>
              </li>
              <li className="flex gap-2">
                <CheckCircle2 size={16} className="shrink-0 mt-0.5" />
                <span>Final clinical decisions must always be made by a certified dermatologist.</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Section 2: Data & Privacy */}
        <TermSection 
          icon={<Lock className="text-[#1E90FF]" />}
          title="2. Data Privacy & Security"
        >
          <p>
            We take patient privacy seriously. While TRACE processes dermoscopic images for analysis:
          </p>
          <ul className="list-disc pl-5 mt-3 space-y-2 text-slate-600 text-sm">
            <li>Users must <strong>NOT</strong> upload images containing identifiable patient faces, tattoos, or ID tags.</li>
            <li>All diagnostic logs (Preprocessing, Fusion, Classification) are stored securely in our MongoDB audit trail.</li>
            <li>We do not sell or share uploaded medical data with third-party advertisers.</li>
          </ul>
        </TermSection>

        {/* Section 3: Usage Guidelines */}
        <TermSection 
          icon={<FileText className="text-[#1E90FF]" />}
          title="3. Acceptable Use"
        >
          <p>
            The system is designed for specific user roles:
          </p>
          <ul className="list-disc pl-5 mt-3 space-y-2 text-slate-600 text-sm">
            <li><strong>Medical Students:</strong> May use TRACE for learning, recognizing patterns (LBP), and understanding AI explainability (Grad-CAM).</li>
            <li><strong>Clinicians:</strong> May use TRACE to validate manual diagnoses or triage cases.</li>
            <li><strong>Restrictions:</strong> You agree not to reverse-engineer the AI models or use the API for non-medical purposes.</li>
          </ul>
        </TermSection>

        {/* Section 4: Limitation of Liability */}
        <TermSection 
          icon={<Scale className="text-[#1E90FF]" />}
          title="4. Limitation of Liability"
        >
          <p>
            The developers of TRACE and the affiliated university make no warranties regarding the absolute accuracy of the hair removal algorithm, feature fusion, or classification outputs. 
            <br/><br/>
            Under no circumstances shall TRACE be liable for any direct, indirect, or consequential damages arising from the use or inability to use the system.
          </p>
        </TermSection>

      </div>

      {/* 4. Footer */}
      <footer className="bg-white border-t border-slate-200 py-10 text-center">
        <div className="max-w-2xl mx-auto px-6">
          <p className="text-slate-500 text-sm mb-4">
            If you have questions about these terms, please contact our administrative team.
          </p>
          <Link to="/contact" className="inline-block bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-6 py-2 rounded-full transition-colors text-sm">
            Contact Support
          </Link>
          <p className="text-xs text-slate-400 mt-8">© 2025 TRACE System. All rights reserved.</p>
        </div>
      </footer>

    </div>
  );
};

// Reusable Section Component
const TermSection = ({ icon, title, children }) => (
  <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-100">
    <div className="flex items-center gap-3 mb-4">
      <div className="bg-blue-50 p-2 rounded-lg">
        {icon}
      </div>
      <h2 className="text-xl font-bold text-slate-800">{title}</h2>
    </div>
    <div className="text-slate-600 leading-relaxed pl-1">
      {children}
    </div>
  </div>
);

export default Terms;