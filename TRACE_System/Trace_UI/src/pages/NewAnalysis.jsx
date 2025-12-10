import React, { useState } from 'react';
import { UploadCloud, CheckCircle2, Scan, ChevronRight, Activity, BrainCircuit, Search, Layers, Download } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import PrimaryButton from '../components/PrimaryButton';

const NewAnalysis = () => {
  const [step, setStep] = useState('upload'); // 'upload' | 'validating' | 'preprocessing' | 'results'
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  
  // Simulation States
  const [validationLogs, setValidationLogs] = useState([]);
  const [prepLogs, setPrepLogs] = useState({ hair: 'pending', features: 'pending' });

  // --- 1. UPLOAD HANDLER ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      startValidation();
    }
  };

  // --- 2. VALIDATION STAGE ---
  const startValidation = () => {
    setStep('validating');
    setValidationLogs([]);

    const checks = [
      { msg: 'Checking Image Format...', delay: 500 },
      { msg: 'Validating Dermoscopic Quality...', delay: 1500 },
      { msg: 'Checking Resolution & Blur...', delay: 2500 },
      { msg: 'Color Range Analysis...', delay: 3500 },
    ];

    let delaySum = 0;
    checks.forEach((check, index) => {
      delaySum += check.delay;
      setTimeout(() => {
        setValidationLogs(prev => [...prev, check.msg]);
        
        if (index === checks.length - 1) {
          setTimeout(() => {
            startPreprocessing();
          }, 1000);
        }
      }, delaySum);
    });
  };

  // --- 3. PREPROCESSING STAGE ---
  const startPreprocessing = () => {
    setStep('preprocessing');
    
    setTimeout(() => {
      setPrepLogs(prev => ({ ...prev, hair: 'completed' }));
    }, 2000);

    setTimeout(() => {
      setPrepLogs(prev => ({ ...prev, features: 'extracting' }));
    }, 3000);

    setTimeout(() => {
      setPrepLogs(prev => ({ ...prev, features: 'completed' }));
      setStep('results');
    }, 5500);
  };

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto">
        
        {/* HEADER */}
        <div className="mb-8 border-b border-slate-200 pb-4">
          <h1 className="text-2xl font-bold text-slate-800">
            {step === 'results' ? 'Final Diagnostic Result' : 'New Analysis Workflow'}
          </h1>
          <p className="text-slate-500 text-sm">
            {step === 'upload' ? 'Step 1: Image Upload' : 
             step === 'validating' ? 'Step 2: Quality Validation' :
             step === 'preprocessing' ? 'Step 3: Preprocessing & Feature Extraction' :
             'Step 4: Classification & Explainability'}
          </p>
        </div>

        {/* --- VIEW 1: UPLOAD --- */}
        {step === 'upload' && (
          <div className="bg-white border-2 border-dashed border-slate-300 rounded-3xl p-16 text-center hover:border-[#1E90FF] transition-colors">
            <div className="w-24 h-24 bg-blue-50 rounded-full flex items-center justify-center mx-auto mb-6 text-[#1E90FF]">
              <UploadCloud size={48} />
            </div>
            <h3 className="text-2xl font-bold text-slate-800">Upload Dermoscopic Image</h3>
            <p className="text-slate-400 mb-8 mt-2 max-w-md mx-auto">
              Required: High-resolution dermoscopic image (.jpg, .png). 
              Non-clinical or blurry images will be rejected by the auto-validator.
            </p>
            <input type="file" id="fileUpload" className="hidden" onChange={handleFileChange} accept="image/*" />
            <PrimaryButton onClick={() => document.getElementById('fileUpload').click()}>
              Select Image From Device
            </PrimaryButton>
          </div>
        )}

        {/* --- VIEW 2: VALIDATION --- */}
        {step === 'validating' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            
            {/* Left: Spinner */}
            <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200 flex flex-col items-center justify-center text-center">
               <div className="relative w-24 h-24 mb-6">
                  <svg className="animate-spin w-full h-full text-blue-200" viewBox="0 0 24 24">
                     <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                     <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Search className="text-[#1E90FF]" />
                  </div>
               </div>
               <h3 className="text-xl font-bold text-slate-800">Validating Image Quality...</h3>
               <p className="text-slate-500 text-sm mt-2">Checking for blur, noise, and artifacts.</p>
            </div>

            {/* Right: Technical Logs (FIXED: &gt; use kiya hai) */}
            <div className="bg-slate-900 text-green-400 p-6 rounded-3xl font-mono text-sm shadow-inner overflow-y-auto h-full min-h-[300px]">
              <p className="border-b border-slate-700 pb-2 mb-2 text-slate-400">System Logs</p>
              <div className="space-y-2">
                <p>&gt; Upload successful</p>
                {validationLogs.map((log, index) => (
                  <p key={index} className="animate-fade-in">&gt; {log}</p>
                ))}
                {validationLogs.length < 4 && (
                   <span className="inline-block w-2 h-4 bg-green-400 animate-pulse"/>
                )}
              </div>
            </div>
          </div>
        )}

        {/* --- VIEW 3: PREPROCESSING --- */}
        {step === 'preprocessing' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Step 1: Hair Removal Visual */}
            <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-slate-800 flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${prepLogs.hair === 'completed' ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`} />
                  Step 1: Hair Removal
                </h3>
                {prepLogs.hair === 'completed' && <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-lg font-bold">COMPLETED</span>}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-slate-500 mb-2 font-semibold text-center">Original</p>
                  <img src={preview} className="rounded-xl w-full aspect-square object-cover border border-slate-200" />
                </div>
                <div>
                  <p className="text-xs text-slate-500 mb-2 font-semibold text-center">Hair Removed</p>
                  {prepLogs.hair === 'completed' ? (
                    <img src={preview} className="rounded-xl w-full aspect-square object-cover border border-green-200 blur-[0.5px] brightness-105 transition-all duration-1000" />
                  ) : (
                    <div className="w-full aspect-square rounded-xl bg-slate-100 flex items-center justify-center text-slate-400 text-xs animate-pulse">
                      Processing...
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Step 2: Feature Extraction Progress */}
            <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex flex-col justify-center">
              <h3 className="font-bold text-slate-800 mb-6 flex items-center gap-2">
                <BrainCircuit className="text-[#1E90FF]" /> Step 2: Feature Pipeline
              </h3>
              
              <div className="space-y-6">
                <FeatureProgress 
                  label="Extracting Texture Features (LBP)" 
                  status={prepLogs.features === 'pending' ? 'waiting' : 'done'} 
                />
                <FeatureProgress 
                  label="Extracting Deep Features (CNN)" 
                  status={prepLogs.features === 'pending' ? 'waiting' : 'done'} 
                />
                <FeatureProgress 
                  label="Fusing Features (Texture + Color)" 
                  status={prepLogs.features === 'completed' ? 'done' : 'waiting'} 
                />
              </div>
            </div>
          </div>
        )}

        {/* --- VIEW 4: FINAL RESULTS --- */}
        {step === 'results' && (
          <div className="space-y-8 animate-fade-in">
            
            {/* A. CLASSIFICATION CARDS */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <ResultCard 
                label="Stage 1: Diagnosis" 
                value="Suspicious" 
                sub="Cancer-Likely Detected" 
                color="bg-red-50 text-red-700 border-red-100" 
              />
              <ResultCard 
                label="Stage 2: Type" 
                value="Melanoma (MEL)" 
                sub="High Risk Lesion" 
                color="bg-orange-50 text-orange-700 border-orange-100" 
              />
              <ResultCard 
                label="AI Confidence" 
                value="94.2%" 
                sub="High Certainty" 
                color="bg-green-50 text-green-700 border-green-100" 
              />
            </div>

            {/* B. VISUALIZATION & XAI */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              
              {/* Localization Box */}
              <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
                <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <Scan className="text-[#1E90FF]" /> Localization
                </h3>
                <div className="relative rounded-xl overflow-hidden group">
                  <img src={preview} className="w-full h-80 object-cover" />
                  <div className="absolute top-[20%] left-[25%] w-[50%] h-[50%] border-4 border-red-500/80 rounded-lg shadow-[0_0_30px_rgba(239,68,68,0.4)] animate-pulse">
                    <span className="absolute -top-8 left-0 bg-red-600 text-white text-xs font-bold px-2 py-1 rounded shadow-sm">
                      Lesion Detected
                    </span>
                  </div>
                </div>
              </div>

              {/* Explainability (XAI) */}
              <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm flex flex-col">
                <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <Layers className="text-[#1E90FF]" /> Explainability (XAI)
                </h3>
                
                {/* 1. Heatmap Placeholder */}
                <div className="flex-1 bg-slate-900 rounded-xl relative overflow-hidden mb-6 group">
                   <img src={preview} className="w-full h-48 object-cover opacity-60 mix-blend-overlay" />
                   <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/40 via-red-500/40 to-yellow-500/10 mix-blend-color-dodge" />
                   <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">Grad-CAM Heatmap</div>
                </div>

                {/* 2. Attribution Bar */}
                <div>
                  <p className="text-sm font-semibold text-slate-700 mb-3">Feature Attribution</p>
                  <div className="flex items-center gap-2 mb-1 text-xs text-slate-500">
                    <span>Texture (LBP)</span>
                    <span className="ml-auto font-bold text-slate-800">62%</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full mb-3 overflow-hidden">
                    <div className="bg-[#1E90FF] h-full" style={{ width: '62%' }}></div>
                  </div>

                  <div className="flex items-center gap-2 mb-1 text-xs text-slate-500">
                    <span>Color (Deep Features)</span>
                    <span className="ml-auto font-bold text-slate-800">38%</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                    <div className="bg-purple-500 h-full" style={{ width: '38%' }}></div>
                  </div>
                </div>
              </div>

            </div>

            {/* C. FOOTER ACTIONS */}
            <div className="flex flex-wrap justify-end gap-4 pt-4 border-t border-slate-200">
              {/* No Download Button (Removed as per previous request) */}
              <button 
                onClick={() => setStep('upload')}
                className="flex items-center gap-2 px-6 py-3 bg-[#1E90FF] text-white rounded-xl font-bold hover:bg-blue-600 transition-all shadow-lg shadow-blue-200"
              >
                Save & Start New
              </button>
            </div>

          </div>
        )}

      </div>
    </DashboardLayout>
  );
};

// --- SUB COMPONENTS ---

const FeatureProgress = ({ label, status }) => (
  <div className="flex items-center gap-3">
    <div className={`w-6 h-6 rounded-full flex items-center justify-center border-2 ${
      status === 'done' ? 'bg-green-500 border-green-500 text-white' : 
      status === 'waiting' ? 'border-slate-200' : 'border-[#1E90FF] border-t-transparent animate-spin'
    }`}>
      {status === 'done' && <CheckCircle2 size={14} />}
    </div>
    <span className={`text-sm font-medium ${status === 'waiting' ? 'text-slate-400' : 'text-slate-700'}`}>
      {label}
    </span>
  </div>
);

const ResultCard = ({ label, value, sub, color }) => (
  <div className={`p-6 rounded-3xl border ${color} text-center`}>
    <p className="text-xs font-bold uppercase tracking-wider opacity-70 mb-2">{label}</p>
    <h2 className="text-3xl font-extrabold mb-1">{value}</h2>
    <p className="text-sm font-medium opacity-80">{sub}</p>
  </div>
);

export default NewAnalysis;