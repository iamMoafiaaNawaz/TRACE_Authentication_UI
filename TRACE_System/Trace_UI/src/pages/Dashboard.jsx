import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { UploadCloud, Activity, Clock, AlertCircle, ArrowRight, Scissors, ShieldAlert } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const Dashboard = () => {
  const [userName, setUserName] = useState('Loading...');
  const [userRole, setUserRole] = useState('User');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUserName(parsedUser.fullName || 'User');
        setUserRole(parsedUser.role || 'User');
      } catch (e) {
        console.error('Failed to parse user:', e);
      }
    }

    const fetchDashboardData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setError('Please login to load dashboard data.');
          return;
        }

        const response = await fetch('http://127.0.0.1:5000/api/history', {
          headers: { Authorization: `Bearer ${token}` }
        });

        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body?.error || 'Failed to load dashboard data');
        }

        const data = await response.json();
        const list = Array.isArray(data) ? data : [];
        const sorted = [...list].sort((a, b) => toEpoch(b) - toEpoch(a));
        setHistory(sorted);
      } catch (fetchError) {
        console.error('Failed to fetch dashboard data:', fetchError);
        setError(fetchError.message || 'Failed to fetch dashboard data.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const stats = useMemo(() => {
    const totalScans = history.length;
    const preprocessed = history.filter((item) => (item?.status || '').toLowerCase() === 'hair_processed_only').length;
    const flagged = history.filter((item) => item?.result === 'Malignant (Cancerous)').length;
    const pending = history.filter((item) => (item?.status || '').toLowerCase() === 'pending').length;
    return { totalScans, preprocessed, flagged, pending };
  }, [history]);

  const recentScans = useMemo(() => history.slice(0, 6), [history]);

  const statusPill = (item) => {
    const status = (item?.status || '').toLowerCase();
    if (item?.result === 'Malignant (Cancerous)') return 'bg-red-50 text-red-700 border-red-100';
    if (status === 'pending') return 'bg-amber-50 text-amber-700 border-amber-100';
    if (status === 'hair_processed_only') return 'bg-blue-50 text-blue-700 border-blue-100';
    return 'bg-green-50 text-green-700 border-green-100';
  };

  return (
    <DashboardLayout>
      <div className="animate-fade-in w-full max-w-full space-y-6 md:space-y-8 overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 w-full">
          <div className="min-w-0">
            <h1 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight truncate">Overview</h1>
            <p className="text-slate-500 mt-1 text-sm truncate">
              Welcome back, <span className="font-semibold text-blue-600">{userName}</span> ({userRole}).
            </p>
          </div>

          <div className="hidden md:block shrink-0">
            <div className="flex items-center gap-2 text-slate-500 text-sm font-medium bg-white px-3 py-1 rounded-full border border-slate-200 shadow-sm">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
              </span>
              Service Online
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-700">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 w-full">
          <StatCard title="Total Records" value={isLoading ? '-' : stats.totalScans} icon={<Activity size={20} />} />
          <StatCard title="Preprocessed" value={isLoading ? '-' : stats.preprocessed} icon={<Scissors size={20} />} />
          <StatCard title="Flagged" value={isLoading ? '-' : stats.flagged} icon={<ShieldAlert size={20} />} />
          <StatCard title="Pending" value={isLoading ? '-' : stats.pending} icon={<Clock size={20} />} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full">
          <div className="lg:col-span-2 bg-blue-600 rounded-2xl p-6 text-white shadow-lg shadow-blue-200 relative overflow-hidden flex flex-col justify-between gap-6">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
            <div className="relative z-10">
              <h2 className="text-xl font-bold mb-2">Run New Hair-Removal Analysis</h2>
              <p className="text-blue-100 text-sm max-w-md leading-relaxed">
                Upload a dermoscopic image to generate hair mask and cleaned output.
              </p>
            </div>
            <Link
              to="/dashboard/upload"
              className="relative z-10 w-full sm:w-auto bg-white text-blue-600 px-6 py-3 rounded-xl font-bold text-sm hover:bg-blue-50 transition-colors flex items-center justify-center gap-2 shadow-sm text-center"
            >
              <UploadCloud size={18} />
              <span>Start Analysis</span>
            </Link>
          </div>

          <div className="lg:col-span-1 bg-white border border-slate-200 rounded-2xl p-6 flex flex-col justify-center items-start shadow-sm h-full w-full">
            <div className="bg-slate-50 p-3 rounded-xl mb-4 text-slate-400">
              <AlertCircle size={24} />
            </div>
            <h2 className="text-lg font-bold text-slate-800 mb-1">History & Exports</h2>
            <p className="text-slate-500 text-sm mb-6">View previous records and download JSON reports.</p>
            <Link to="/dashboard/history" className="text-blue-600 font-semibold text-sm flex items-center gap-1 hover:gap-2 transition-all">
              Open History <ArrowRight size={16} />
            </Link>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm w-full">
          <div className="px-6 py-5 border-b border-slate-100 flex items-center justify-between">
            <h3 className="text-base font-bold text-slate-800">Recent Activity</h3>
            <Link to="/dashboard/history" className="text-xs font-semibold text-blue-600 hover:underline">View all</Link>
          </div>

          <div className="overflow-x-auto w-full">
            <table className="w-full text-left min-w-[760px]">
              <thead className="bg-slate-50 text-slate-500 text-xs font-bold uppercase tracking-wider">
                <tr>
                  <th className="px-6 py-4">Date</th>
                  <th className="px-6 py-4">Diagnosis</th>
                  <th className="px-6 py-4">Confidence</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {isLoading ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center text-slate-400 text-sm">
                      <span className="inline-flex items-center gap-2">
                        <span className="w-4 h-4 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"></span>
                        Loading recent activity...
                      </span>
                    </td>
                  </tr>
                ) : recentScans.length > 0 ? (
                  recentScans.map((item) => (
                    <tr key={item._id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4 text-sm text-slate-600">{item?.date || '-'}</td>
                      <td className="px-6 py-4 text-sm text-slate-800 font-semibold">{item?.diagnosis || item?.result || '-'}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">{item?.confidence || '-'}</td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold border ${statusPill(item)}`}>
                          {item?.status || 'completed'}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <Link to="/dashboard/history" className="text-blue-600 text-sm font-semibold hover:underline">
                          View
                        </Link>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center text-slate-400 text-sm">
                      No recent activity found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

const StatCard = ({ title, value, icon }) => (
  <div className="bg-white p-5 rounded-2xl border border-slate-200 flex items-center gap-4 shadow-sm w-full overflow-hidden">
    <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center text-slate-500 border border-slate-100 shrink-0">
      {icon}
    </div>
    <div className="flex-1 min-w-0">
      <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider truncate">{title}</p>
      <h3 className="text-2xl font-bold text-slate-800 mt-0.5 truncate">{value}</h3>
    </div>
  </div>
);

export default Dashboard;
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
