import React, { useEffect, useMemo, useState } from 'react';
import { Download, Calendar, Search, AlertCircle, Activity, ShieldAlert, CheckCircle2 } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import axios from 'axios';

const statusMeta = (item) => {
  const status = (item?.status || '').toLowerCase();
  const isFlagged = item?.result === 'Malignant (Cancerous)';

  if (isFlagged) return { label: 'Flagged', className: 'bg-red-50 text-red-700 border-red-100' };
  if (status === 'hair_processed_only') return { label: 'Preprocessed', className: 'bg-blue-50 text-blue-700 border-blue-100' };
  if (status === 'pending') return { label: 'Pending', className: 'bg-amber-50 text-amber-700 border-amber-100' };
  return { label: 'Normal', className: 'bg-green-50 text-green-700 border-green-100' };
};

const safeText = (value, fallback = '-') => (value === null || value === undefined || value === '' ? fallback : String(value));
const toEpoch = (item) => {
  if (typeof item?.created_at_epoch === 'number' && item.created_at_epoch > 0) return item.created_at_epoch;
  if (item?.created_at_iso) {
    const t = Date.parse(item.created_at_iso);
    return Number.isNaN(t) ? 0 : Math.floor(t / 1000);
  }
  if (item?.date) {
    const t = Date.parse(item.date);
    return Number.isNaN(t) ? 0 : Math.floor(t / 1000);
  }
  return 0;
};

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setHistory([]);
          setError('Please login to view history.');
          return;
        }

        const res = await axios.get('http://127.0.0.1:5000/api/history', {
          headers: { Authorization: `Bearer ${token}` }
        });
        const list = Array.isArray(res.data) ? res.data : [];
        const sorted = [...list].sort((a, b) => toEpoch(b) - toEpoch(a));
        setHistory(sorted);
      } catch (err) {
        console.error('Failed to fetch history:', err);
        setHistory([]);
        setError(err?.response?.data?.error || 'Unable to load history records.');
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const filteredHistory = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    if (!q) return history;

    return history.filter((item) => {
      const hay = [
        safeText(item?.date, '').toLowerCase(),
        safeText(item?.diagnosis, '').toLowerCase(),
        safeText(item?.result, '').toLowerCase(),
        safeText(item?.confidence, '').toLowerCase(),
        safeText(item?.status, '').toLowerCase()
      ];
      return hay.some((txt) => txt.includes(q));
    });
  }, [history, searchTerm]);

  const total = history.length;
  const flagged = history.filter((item) => item?.result === 'Malignant (Cancerous)').length;
  const preprocessed = history.filter((item) => (item?.status || '').toLowerCase() === 'hair_processed_only').length;

  const downloadRecord = (item) => {
    const blob = new Blob([JSON.stringify(item, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trace-record-${item?._id || 'record'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <DashboardLayout>
      <div className="w-full max-w-6xl mx-auto font-sans animate-fade-in space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight">Analysis History</h1>
            <p className="text-slate-500 text-sm mt-1">Review and export your previous scan records.</p>
          </div>

          <div className="relative w-full lg:w-80">
            <Search className="absolute left-3 top-2.5 text-slate-400" size={18} />
            <input
              type="text"
              placeholder="Search by date, diagnosis, status..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm outline-none focus:border-[#1E90FF] focus:ring-2 focus:ring-blue-50 transition-all shadow-sm"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <SummaryCard icon={<Activity size={18} />} label="Total Records" value={total} />
          <SummaryCard icon={<CheckCircle2 size={18} />} label="Preprocessed" value={preprocessed} />
          <SummaryCard icon={<ShieldAlert size={18} />} label="Flagged" value={flagged} />
        </div>

        {error && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-700">
            {error}
          </div>
        )}

        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm w-full">
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left min-w-[760px]">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 text-xs font-bold uppercase tracking-wider">
                <tr>
                  <th className="px-6 py-4 whitespace-nowrap">Date Scanned</th>
                  <th className="px-6 py-4 whitespace-nowrap">Diagnosis</th>
                  <th className="px-6 py-4 whitespace-nowrap">Result</th>
                  <th className="px-6 py-4 whitespace-nowrap">Confidence</th>
                  <th className="px-6 py-4 whitespace-nowrap">Status</th>
                  <th className="px-6 py-4 text-right whitespace-nowrap">Export</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {loading ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-12 text-center text-slate-400">
                      <div className="flex justify-center items-center gap-2">
                        <span className="w-4 h-4 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"></span>
                        Loading records...
                      </div>
                    </td>
                  </tr>
                ) : filteredHistory.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-12 text-center">
                      <div className="flex flex-col items-center justify-center text-slate-400">
                        <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mb-3">
                          <AlertCircle size={24} />
                        </div>
                        <p className="text-sm font-medium text-slate-600">No history found</p>
                        <p className="text-xs mt-1">Try another search term or run a new analysis.</p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredHistory.map((item) => {
                    const meta = statusMeta(item);
                    return (
                      <tr key={item._id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="px-6 py-4 text-sm text-slate-600 whitespace-nowrap">
                          <span className="inline-flex items-center gap-2">
                            <Calendar size={16} className="text-slate-400" />
                            {safeText(item?.date)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm font-semibold text-slate-800 whitespace-nowrap">
                          {safeText(item?.diagnosis)}
                        </td>
                        <td className="px-6 py-4 text-sm text-slate-600 whitespace-nowrap">
                          {safeText(item?.result)}
                        </td>
                        <td className="px-6 py-4 text-sm text-slate-600 whitespace-nowrap">
                          {safeText(item?.confidence)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold border ${meta.className}`}>
                            {meta.label}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right whitespace-nowrap">
                          <button
                            onClick={() => downloadRecord(item)}
                            className="inline-flex items-center gap-2 text-slate-500 hover:text-[#1E90FF] transition-colors p-1 hover:bg-blue-50 rounded-lg"
                            title="Download JSON record"
                          >
                            <Download size={16} />
                            <span className="text-xs font-semibold">JSON</span>
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

const SummaryCard = ({ icon, label, value }) => (
  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
    <div className="flex items-center gap-2 text-slate-500 mb-1">
      {icon}
      <p className="text-xs font-bold uppercase tracking-wider">{label}</p>
    </div>
    <p className="text-2xl font-bold text-slate-800">{value}</p>
  </div>
);

export default History;
