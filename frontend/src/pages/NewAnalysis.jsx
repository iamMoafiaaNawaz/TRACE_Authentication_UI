import React, { useEffect, useState } from 'react';
import { UploadCloud, ImagePlus, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import PrimaryButton from '../components/PrimaryButton';
import { submitAnalysis } from '../services/analysisService';

const CLASS_META = {
  MEL: { code: 'MEL', label: 'Melanoma', severity: 'Critical', kind: 'Malignant' },
  BCC: { code: 'BCC', label: 'Basal Cell Carcinoma', severity: 'High', kind: 'Malignant' },
  BKL: { code: 'BKL', label: 'Benign Keratosis', severity: 'Low', kind: 'Benign' },
  NV: { code: 'NV', label: 'Nevus (Mole)', severity: 'Low', kind: 'Benign' },
};

const malignantCodes = new Set(['MEL', 'BCC']);

const clamp01 = (v) => Math.min(1, Math.max(0, v));

const formatPct = (v01, decimals = 2) => `${(clamp01(v01) * 100).toFixed(decimals)}%`;

const toneStyles = (tone) => {
  if (tone === 'danger') {
    return {
      ring: 'stroke-red-600',
      accent: 'text-red-700',
      badge: 'border-red-200 text-red-700 bg-white',
    };
  }
  return {
    ring: 'stroke-emerald-600',
    accent: 'text-emerald-700',
    badge: 'border-emerald-200 text-emerald-700 bg-white',
  };
};

const firstDetection = (localization) => {
  if (!localization) return null;
  if (Array.isArray(localization?.detections) && localization.detections.length > 0) {
    return localization.detections[0];
  }
  if (Array.isArray(localization?.bbox_normalized) && localization.bbox_normalized.length === 4) {
    return {
      bbox_normalized: localization.bbox_normalized,
      confidence: localization?.confidence ?? null,
      label: localization?.label ?? null,
      class_id: localization?.class_id ?? null,
    };
  }
  return null;
};

const bboxStyleFromLocalization = (localization) => {
  if (localization?.status !== 'success') return null;
  if (localization?.box_found !== true) return null;
  const det = firstDetection(localization);
  const bbox = det?.bbox_normalized;
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;

  const [x1, y1, x2, y2] = bbox.map((v) => Number(v));
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return null;

  // Clamp coordinates then compute CSS percentages.
  const xmin = clamp01(x1);
  const ymin = clamp01(y1);
  const xmax = clamp01(x2);
  const ymax = clamp01(y2);

  const left = xmin * 100;
  const top = ymin * 100;
  const width = Math.max(0, (xmax - xmin) * 100);
  const height = Math.max(0, (ymax - ymin) * 100);
  return {
    left: `${left}%`,
    top: `${top}%`,
    width: `${width}%`,
    height: `${height}%`,
    zIndex: 10,
  };
};

const lesionCoveragePct = (localization) => {
  const det = firstDetection(localization);
  const bbox = det?.bbox_normalized;
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const [x1, y1, x2, y2] = bbox.map((v) => Number(v));
  if (![x1, y1, x2, y2].every((v) => Number.isFinite(v))) return null;
  const xmin = clamp01(x1);
  const ymin = clamp01(y1);
  const xmax = clamp01(x2);
  const ymax = clamp01(y2);
  const w = Math.max(0, xmax - xmin);
  const h = Math.max(0, ymax - ymin);
  return (w * h) * 100;
};

const NewAnalysis = () => {
  const [step, setStep] = useState('upload'); // upload | processing | results
  const [preview, setPreview] = useState(null);
  const [processedPreview, setProcessedPreview] = useState(null);
  const [apiResult, setApiResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showLocalization, setShowLocalization] = useState(true);
  const [progress, setProgress] = useState(0);

  const progressLabel = (p) => {
    if (p < 20) return 'Uploading Image...';
    if (p < 50) return 'Applying Hair Removal...';
    if (p < 80) return 'Running ConvNeXt Classification...';
    if (p < 90) return 'Running YOLO Localization...';
    if (p < 100) return 'Finalizing results...';
    return 'Completed.';
  };

  useEffect(() => {
    if (step !== 'processing') return undefined;

    let canceled = false;
    const id = setInterval(() => {
      if (canceled) return;
      setProgress((prev) => {
        let stepSize = 1;
        if (prev < 20) stepSize = 3;
        else if (prev < 50) stepSize = 2;
        else if (prev < 80) stepSize = 1;
        else stepSize = 0.5;
        const p = Math.min(90, prev + stepSize);
        return p;
      });
    }, 700);

    return () => {
      canceled = true;
      clearInterval(id);
    };
  }, [step]);

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
    setShowLocalization(true);
    setProgress(0);
    setStep('processing');

    try {
      const data = await submitAnalysis(selectedFile);
      setApiResult(data);
      setProcessedPreview(data?.processed_image || null);
      setProgress(100);
      await new Promise((resolve) => setTimeout(resolve, 180));
      setStep('results');
    } catch (err) {
      setError(err.message || 'Failed to process image');
      setStep('upload');
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-8 font-sans">
        <header className="pt-1">
          <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight">
            Skin Cancer Detection & Analysis
          </h1>
          <p className="text-slate-500 text-sm mt-2 leading-relaxed">
            Upload a dermoscopic image. The system performs hair-removal preprocessing, runs ConvNeXt classification, and optionally localizes the lesion.
          </p>
        </header>

        {error && (
          <div className="bg-white text-slate-900 p-5 rounded-2xl border border-slate-200 flex items-start gap-3">
            <AlertCircle size={20} className="text-red-700 mt-0.5 shrink-0" />
            <div className="min-w-0">
              <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Issue</p>
              <p className="text-sm text-slate-700 mt-1">{error}</p>
            </div>
          </div>
        )}

        {step === 'upload' && (
          <section
            className={`rounded-3xl border transition-all ${
              isDragging ? 'border-slate-400 bg-slate-50' : 'border-slate-200 bg-white'
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
            <div className="p-10 md:p-12 text-center">
              <div className="w-20 h-20 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center mx-auto text-slate-700 mb-6">
                <UploadCloud size={40} />
              </div>
              <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Upload Dermoscopic Image</h2>
              <p className="text-slate-500 max-w-lg mx-auto mt-2 mb-8 text-sm leading-relaxed">
                Supported formats: JPG, JPEG, PNG. Results include classification output plus preprocessing artifacts.
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
              <p className="mt-6 text-[11px] font-bold uppercase tracking-widest text-slate-400">
                MEL - BCC - BKL - NV
              </p>
            </div>
          </section>
        )}

        {step === 'processing' && (
          <section className="bg-white border border-slate-200 rounded-3xl p-10">
            <div className="flex items-center justify-between gap-4 text-slate-700 mb-4">
              <div className="flex items-center gap-3">
                <Sparkles size={18} className="text-slate-500" />
                <p className="font-semibold">{progressLabel(progress)}</p>
              </div>
              <p className="text-sm font-bold tabular-nums text-slate-900">{progress}%</p>
            </div>
            <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-slate-900 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <p className="mt-4 text-xs text-slate-500 leading-relaxed">
              Hair removal is applied before classification. Localization runs on the same processed image.
            </p>
          </section>
        )}

        {step === 'results' && (
          <section className="space-y-6">
            <ResultHero 
              apiResult={apiResult} 
              processedPreview={processedPreview} 
              showLocalization={showLocalization} 
              onToggleLocalization={() => setShowLocalization((v) => !v)} 
            /> 

            <PreprocessingSection 
              preview={preview} 
              processedPreview={processedPreview} 
              apiResult={apiResult} 
            /> 

            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2 text-slate-500 text-sm">
                <CheckCircle2 size={16} className="text-emerald-600" />
                <span>{apiResult?.message || 'Analysis completed.'}</span>
              </div>
              <button
                onClick={() => {
                  setStep('upload');
                  setPreview(null);
                  setProcessedPreview(null);
                  setApiResult(null);
                  setError(null);
                  setShowLocalization(true);
                }}
                className="px-5 py-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white font-semibold"
              >
                New Analysis
              </button>
            </div>
          </section>
        )}
      </div>
    </DashboardLayout>
  );
};

const ResultHero = ({ apiResult, processedPreview, showLocalization, onToggleLocalization }) => {
  const classification = apiResult?.classification || null;
  const code = classification?.class_code || null;
  const known = code && CLASS_META[code] ? CLASS_META[code] : null;

  const diagnosis = classification?.diagnosis || apiResult?.diagnosis || known?.label || '-';
  const kind = classification?.result || apiResult?.result || known?.kind || '-';
  const severity = classification?.severity || apiResult?.severity || known?.severity || 'N/A';

  const confidenceFloat =
    typeof classification?.confidence === 'number'
      ? classification.confidence
      : (() => {
          const raw = apiResult?.confidence;
          if (typeof raw === 'number') return raw > 1 ? raw / 100 : raw;
          if (typeof raw === 'string') {
            const n = Number(raw.replace('%', '').trim());
            if (Number.isFinite(n)) return n > 1 ? n / 100 : n;
          }
          return null;
        })();
  const confidence01 = clamp01(confidenceFloat ?? 0);

  const isMalignant = code ? malignantCodes.has(code) : String(kind || '').toLowerCase().includes('malignant');
  const tone = isMalignant ? 'danger' : 'ok';
  const styles = toneStyles(tone);

  const loc = apiResult?.localization || null;
  const locDet = firstDetection(loc);
  const locFound = loc?.status === 'success' && loc?.box_found === true && Array.isArray(locDet?.bbox_normalized) && locDet.bbox_normalized.length === 4;
  const locConf = typeof locDet?.confidence === 'number' ? locDet.confidence : null;
  const locLabel = typeof locDet?.label === 'string' && locDet.label ? locDet.label : null;
  const heroOverlayStyle = showLocalization ? bboxStyleFromLocalization(loc) : null;
  const coveragePct = lesionCoveragePct(loc);
  const heroImageSrc = processedPreview || apiResult?.processed_image || null;

  const recommendation =
    code === 'MEL'
      ? 'Requires immediate consultation with a dermatologist.'
      : code === 'BCC'
      ? 'High-priority specialist review is recommended.'
      : code === 'BKL'
      ? 'Likely benign; routine clinical follow-up advised.'
      : code === 'NV'
      ? 'Likely benign nevus; monitor for visible changes.'
      : 'Clinical review is recommended to confirm this prediction.';

  const detailType = isMalignant ? 'Malignant' : 'Benign';
  const typeTone = isMalignant ? 'text-red-700 bg-red-50 border-red-200' : 'text-emerald-700 bg-emerald-50 border-emerald-200';

  return (
    <div className="bg-white border border-slate-200 rounded-3xl shadow-sm">
      <div className="p-8 md:p-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start">
          <div className="min-w-0 space-y-6">
            <div className="flex flex-wrap items-center gap-2">
              <Badge>SKIN CANCER CLASSIFICATION</Badge>
              {code && <Badge>CLASS {code}</Badge>}
              <Badge className={`${styles.badge} border`}>{String(kind).toUpperCase()}</Badge>
              <Badge>SEVERITY {String(severity).toUpperCase()}</Badge>
            </div>

            <div className="space-y-2">
              <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Detailed Diagnosis</p>
              <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 tracking-tight">{diagnosis}</h2>
              <p className="text-sm text-slate-500 leading-relaxed">
                This output is a model prediction. For clinical decisions, consult a licensed dermatologist.
              </p>
            </div>

            <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
              <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500 mb-4">Medical Pathology Report</p>
              <div className="space-y-3 text-sm text-slate-700">
                <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="font-bold text-slate-800">Detailed Diagnosis</span>
                  <span className="text-right">{diagnosis}</span>
                </div>
                <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="font-bold text-slate-800">Type</span>
                  <span className={`inline-flex items-center px-2.5 py-1 rounded-full border text-xs font-bold ${typeTone}`}>{detailType}</span>
                </div>
                <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="font-bold text-slate-800">Probability Score</span>
                  <span className="text-lg font-extrabold text-slate-900">{formatPct(confidence01, 1)}</span>
                </div>
                <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="font-bold text-slate-800">Severity Level</span>
                  <span>{severity}</span>
                </div>
                <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2">
                  <span className="font-bold text-slate-800">Class Code</span>
                  <span>{code || 'N/A'}</span>
                </div>
                <div className="pt-1">
                  <span className="font-bold text-slate-800">Recommendation: </span>
                  <span className="text-slate-700">{recommendation}</span>
                </div>
              </div>
            </div>

            <ConfidenceGauge value01={confidence01} tone={tone} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Processed Result Image</p>
                <p className="text-xs text-slate-600 mt-1 font-semibold">YOLO: {loc?.status || 'N/A'}</p>
              </div>
              {locFound && (
                <button
                  type="button"
                  onClick={onToggleLocalization}
                  className="shrink-0 inline-flex items-center px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-widest border border-slate-200 text-slate-700 bg-white hover:bg-slate-50"
                  title="Toggle localization overlay"
                >
                  {showLocalization ? 'Hide Box' : 'Show Box'}
                </button>
              )}
            </div>

            <div className="relative w-full rounded-2xl border border-slate-200 bg-white overflow-visible">
              {heroImageSrc ? (
                <>
                  <img src={heroImageSrc} className="block w-full h-auto max-h-none object-contain" alt="Processed result" />
                  {heroOverlayStyle && (
                    <div className="absolute border-2 border-red-600 rounded-sm pointer-events-none" style={heroOverlayStyle}>
                      {locFound && (
                        <div className="absolute -top-5 left-0 text-white text-[11px] font-semibold bg-red-600 px-2 py-0.5 rounded-md shadow-sm">
                          {locConf !== null ? `Score: ${formatPct(locConf, 0)}` : 'Score: N/A'}
                          {coveragePct !== null ? ` | Coverage ${coveragePct.toFixed(2)}%` : ''}
                        </div>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="p-8 text-sm text-slate-500">Processed image unavailable.</div>
              )}
            </div>

            {locFound && locLabel && (
              <p className="text-xs text-slate-600">
                Detected label: <span className="font-semibold text-slate-800">{String(locLabel).toUpperCase()}</span>
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const Badge = ({ children, className = '' }) => (
  <span
    className={`inline-flex items-center px-3 py-1 rounded-full text-[11px] font-bold uppercase tracking-widest border border-slate-200 text-slate-700 bg-white ${className}`}
  >
    {children}
  </span>
);

const ConfidenceGauge = ({ value01, tone }) => {
  const v = clamp01(value01);
  const size = 132;
  const stroke = 8;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = c * v;
  const gap = c - dash;
  const styles = toneStyles(tone);

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-3xl p-6 md:p-7">
      <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500 mb-4">Confidence</p>
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle cx={size / 2} cy={size / 2} r={r} strokeWidth={stroke} className="stroke-slate-200" fill="none" />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            strokeWidth={stroke}
            className={styles.ring}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${gap}`}
          />
        </svg>
        <div className="absolute text-center">
          <p className="text-3xl font-extrabold text-slate-900 tabular-nums">{Math.round(v * 100)}</p>
          <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">/ 100</p>
        </div>
      </div>
      <div className="mt-5">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Probability</p>
          <p className={`text-sm font-bold tabular-nums ${styles.accent}`}>{formatPct(v)}</p>
        </div>
        <div className="mt-2 h-2 w-full rounded-full bg-slate-200 overflow-hidden">
          <div className="h-full bg-slate-900" style={{ width: `${Math.round(v * 100)}%` }} />
        </div>
      </div>
    </div>
  );
};

const PreprocessingSection = ({ preview, processedPreview, apiResult }) => {
  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-6 md:p-8 shadow-sm space-y-6">
      <div>
        <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500">Preprocessing Pipeline</p>
        <h3 className="text-lg md:text-xl font-extrabold text-slate-900 mt-2">Hair Removal</h3>
        <p className="text-slate-500 text-sm mt-2 leading-relaxed">
          These artifacts are generated during preprocessing (hair removal). YOLO localization is rendered only on the final processed result image above.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <MiniImageCard title="Original" src={preview} />
        <MiniImageCard title="Mask Overlay" src={apiResult?.mask_overlay_image || preview} />
        <MiniImageCard title="Binary Mask" src={apiResult?.mask_image || preview} />
        <MiniImageCard
          title="Processed"
          src={processedPreview || preview}
        />
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5">
        <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500 mb-3">Hair Removal Details</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-slate-700">
          <p><span className="font-semibold">Method:</span> {apiResult?.hair_removal?.method || 'N/A'}</p>
          <p><span className="font-semibold">Mask Coverage:</span> {apiResult?.hair_removal?.mask_coverage_percent ?? 'N/A'}%</p>
          <p><span className="font-semibold">Backend:</span> {apiResult?.hair_removal?.model_backend || 'N/A'}</p>
          <p><span className="font-semibold">Validation:</span> {apiResult?.hair_removal?.validation_mode || 'N/A'}</p>
        </div>
        {apiResult?.hair_removal?.model_error && (
          <p className="mt-3 text-xs text-slate-600 bg-white border border-slate-200 rounded-xl px-3 py-2">
            Runtime note: {apiResult.hair_removal.model_error}
          </p>
        )}
      </div>
    </div>
  );
};

const MiniImageCard = ({ title, src }) => (
  <div className="bg-white rounded-2xl border border-slate-200 p-4">
    <p className="text-[11px] font-bold uppercase tracking-widest text-slate-500 mb-2">{title}</p>
    <div className="relative inline-block w-full rounded-xl border border-slate-200 overflow-hidden bg-slate-50">
      <img src={src} className="block w-full h-auto object-contain" alt={title} />
    </div>
  </div>
);

export default NewAnalysis;

