import React, { useState } from 'react';
import { UploadCloud, ImagePlus, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import PrimaryButton from '../components/PrimaryButton';

const NewAnalysis = () => {
  const [step, setStep] = useState('upload'); // upload | processing | results
  const [preview, setPreview] = useState(null);
  const [processedPreview, setProcessedPreview] = useState(null);
  const [apiResult, setApiResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const processSelectedFile = async (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.type?.startsWith('image/')) {
      setError('Please provide a valid image file.');
      return;
    }

    setPreview(URL.createObjectURL(selectedFile));
    setProcessedPreview(null);
    setApiResult(null);
    setError(null);
    setStep('processing');

    try {
      const data = await fetchPrediction(selectedFile);
      setApiResult(data);
      setProcessedPreview(data?.processed_image || null);
      setStep('results');
    } catch (err) {
      setError(err.message || 'Failed to process image');
      setStep('upload');
    }
  };

  const fetchPrediction = async (imageFile) => {
    const formData = new FormData();
    formData.append('file', imageFile);
    const token = localStorage.getItem('token');

    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data?.error || 'Server Error');
    return data;
  };

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="mb-4">
          <h1 className="text-3xl font-extrabold text-slate-800">Hair Removal Analysis</h1>
          <p className="text-slate-500 text-sm mt-1">
            Upload or drag-drop a dermoscopic image for hair-removal preprocessing.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 text-red-600 p-4 rounded-xl flex items-center gap-3 border border-red-200">
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        {step === 'upload' && (
          <div
            className={`relative overflow-hidden rounded-3xl border transition-all ${
              isDragging ? 'border-blue-400 bg-blue-50' : 'border-slate-200 bg-white'
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={async (e) => {
              e.preventDefault();
              setIsDragging(false);
              const dropped = e.dataTransfer.files?.[0];
              await processSelectedFile(dropped);
            }}
          >
            <div className="absolute -top-24 -right-24 w-72 h-72 bg-blue-200/30 rounded-full blur-3xl" />
            <div className="absolute -bottom-24 -left-20 w-72 h-72 bg-cyan-100/40 rounded-full blur-3xl" />

            <div className="relative z-10 p-14 text-center">
              <div className="w-24 h-24 rounded-full bg-white shadow-lg shadow-blue-100 flex items-center justify-center mx-auto text-blue-600 mb-6">
                <UploadCloud size={46} />
              </div>
              <h3 className="text-2xl font-bold text-slate-800">Upload or Paste Dermoscopic Image</h3>
              <p className="text-slate-500 max-w-lg mx-auto mt-2 mb-8">
                Supported formats: JPG, JPEG, PNG. The system will generate a clean hair-removed output for preprocessing.
              </p>

              <div className="flex flex-wrap justify-center gap-3">
                <input
                  type="file"
                  id="fileUpload"
                  className="hidden"
                  onChange={(e) => processSelectedFile(e.target.files?.[0])}
                  accept="image/*"
                />
                <PrimaryButton onClick={() => document.getElementById('fileUpload').click()}>
                  <span className="flex items-center gap-2">
                    <ImagePlus size={18} /> Select Image
                  </span>
                </PrimaryButton>
              </div>
            </div>
          </div>
        )}

        {step === 'processing' && (
          <div className="bg-white border border-slate-200 rounded-3xl p-10">
            <div className="flex items-center gap-3 text-blue-600 mb-3">
              <Sparkles size={18} />
              <p className="font-semibold">Generating hair mask and removing hair...</p>
            </div>
            <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 animate-pulse" style={{ width: '85%' }} />
            </div>
          </div>
        )}

        {step === 'results' && (
          <div className="space-y-6">
            <div className="bg-green-50 border border-green-200 rounded-2xl px-5 py-4 text-green-700 text-sm flex items-center gap-2">
              <CheckCircle2 size={18} />
              {apiResult?.message || 'Hair removal completed.'}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm">
                <p className="text-xs text-slate-500 mb-2 font-semibold">With Hair (Original)</p>
                <img src={preview} className="rounded-2xl w-full aspect-square object-cover border border-slate-200" alt="With Hair" />
              </div>

              <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm">
                <p className="text-xs text-slate-500 mb-2 font-semibold">Mask Overlay</p>
                <img
                  src={apiResult?.mask_overlay_image || preview}
                  className="rounded-2xl w-full aspect-square object-cover border border-amber-200"
                  alt="Mask Overlay"
                />
              </div>

              <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm">
                <p className="text-xs text-slate-500 mb-2 font-semibold">Binary Mask Result</p>
                <img
                  src={apiResult?.mask_image || preview}
                  className="rounded-2xl w-full aspect-square object-cover border border-indigo-200 bg-slate-50"
                  alt="Binary Mask"
                />
              </div>

              <div className="bg-white p-5 rounded-3xl border border-slate-200 shadow-sm">
                <p className="text-xs text-slate-500 mb-2 font-semibold">Without Hair (Processed)</p>
                <img
                  src={processedPreview || preview}
                  className="rounded-2xl w-full aspect-square object-cover border border-green-200"
                  alt="Without Hair"
                />
              </div>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="space-y-2">
                <p className="text-xs text-slate-500 uppercase tracking-wider">Pipeline Details</p>
                <p className="text-slate-800 font-bold">
                  Method: {apiResult?.hair_removal?.method || 'N/A'}
                </p>
                <p className="text-sm text-slate-600">
                  Mask Coverage: {apiResult?.hair_removal?.mask_coverage_percent ?? 'N/A'}%
                </p>
                <p className="text-sm text-slate-600">
                  Backend: {apiResult?.hair_removal?.model_backend || 'N/A'}
                </p>
                <p className="text-sm text-slate-600">
                  Validation Mode: {apiResult?.hair_removal?.validation_mode || 'N/A'}
                </p>
                {apiResult?.hair_removal?.model_error && (
                  <p className="text-xs text-blue-700 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
                    Runtime note: {apiResult.hair_removal.model_error}
                  </p>
                )}
              </div>
              <button
                onClick={() => {
                  setStep('upload');
                  setPreview(null);
                  setProcessedPreview(null);
                  setApiResult(null);
                  setError(null);
                }}
                className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold"
              >
                Start New
              </button>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default NewAnalysis;
