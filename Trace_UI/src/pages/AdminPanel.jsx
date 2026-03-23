import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Users,
  Trash2,
  Activity,
  ShieldAlert,
  Search,
  Stethoscope,
  Cpu,
  FileText,
  AlertTriangle,
  Scissors,
  RefreshCw
} from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const api = 'http://127.0.0.1:5000';

const AdminPanel = () => {
  const [users, setUsers] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [stats, setStats] = useState({
    totalUsers: 0, students: 0, clinicians: 0, admins: 0, totalAnalyses: 0, hairOnly: 0, flagged: 0
  });
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searchUsers, setSearchUsers] = useState('');
  const [searchAnalyses, setSearchAnalyses] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const authConfig = () => {
    const token = localStorage.getItem('token');
    return { headers: token ? { Authorization: `Bearer ${token}` } : {} };
  };

  const fetchAdminData = async (showRefresh = false) => {
    try {
      setError('');
      showRefresh ? setRefreshing(true) : setLoading(true);

      const userRaw = localStorage.getItem('user');
      const role = userRaw ? JSON.parse(userRaw)?.role : '';
      if (role !== 'Admin') {
        navigate('/dashboard', { replace: true });
        return;
      }

      const cfg = authConfig();
      const [usersRes, statsRes, analysesRes, statusRes] = await Promise.all([
        axios.get(`${api}/api/admin/users`, cfg),
        axios.get(`${api}/api/admin/analytics`, cfg),
        axios.get(`${api}/api/admin/analyses?limit=200`, cfg),
        axios.get(`${api}/api/admin/system-status`, cfg)
      ]);

      setUsers(Array.isArray(usersRes.data) ? usersRes.data : []);
      setStats(statsRes.data || {});
      setAnalyses(Array.isArray(analysesRes.data) ? analysesRes.data : []);
      setSystemStatus(statusRes.data || null);
    } catch (e) {
      console.error('Admin panel load error:', e);
      const statusCode = e?.response?.status;
      const msg = e?.response?.data?.error || 'Failed to load admin panel data.';
      setError(msg);
      if (statusCode === 401 || statusCode === 403) navigate('/login', { replace: true });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAdminData();
  }, []);

  const handleDeleteUser = async (userId) => {
    if (!window.confirm('Delete this user permanently?')) return;
    try {
      await axios.delete(`${api}/api/admin/users/${userId}`, authConfig());
      setUsers((prev) => prev.filter((u) => u._id !== userId));
    } catch (e) {
      alert(e?.response?.data?.error || 'Failed to delete user.');
    }
  };

  const handleDeleteAnalysis = async (analysisId) => {
    if (!window.confirm('Delete this analysis record permanently?')) return;
    try {
      await axios.delete(`${api}/api/admin/analyses/${analysisId}`, authConfig());
      setAnalyses((prev) => prev.filter((r) => r._id !== analysisId));
    } catch (e) {
      alert(e?.response?.data?.error || 'Failed to delete analysis.');
    }
  };

  const filteredUsers = useMemo(() => {
    const q = searchUsers.trim().toLowerCase();
    if (!q) return users;
    return users.filter((u) =>
      (u?.fullName || '').toLowerCase().includes(q) ||
      (u?.email || '').toLowerCase().includes(q) ||
      (u?.role || '').toLowerCase().includes(q)
    );
  }, [users, searchUsers]);

  const filteredAnalyses = useMemo(() => {
    const q = searchAnalyses.trim().toLowerCase();
    if (!q) return analyses;
    return analyses.filter((r) =>
      (r?.diagnosis || '').toLowerCase().includes(q) ||
      (r?.result || '').toLowerCase().includes(q) ||
      (r?.status || '').toLowerCase().includes(q) ||
      (r?.date || '').toLowerCase().includes(q)
    );
  }, [analyses, searchAnalyses]);

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto animate-fade-in p-2 space-y-8">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-4 border-b border-slate-200 pb-6">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 tracking-tight">Admin Control Center</h1>
            <p className="text-slate-500 mt-1">Full access to users, analyses, and runtime configuration status.</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => fetchAdminData(true)}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
              Refresh
            </button>
            <div className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-bold border border-slate-200">
              <ShieldAlert size={16} className="text-blue-600" /> Administrator Access
            </div>
          </div>
        </div>

        {error && (
          <div className="px-4 py-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-700">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          <StatCard icon={<Users size={18} />} label="Total Users" value={stats.totalUsers ?? 0} />
          <StatCard icon={<Stethoscope size={18} />} label="Clinicians" value={stats.clinicians ?? 0} />
          <StatCard icon={<FileText size={18} />} label="Analyses" value={stats.totalAnalyses ?? 0} />
          <StatCard icon={<AlertTriangle size={18} />} label="Flagged" value={stats.flagged ?? 0} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between gap-3">
              <h2 className="font-bold text-slate-800 flex items-center gap-2"><Users size={18} className="text-blue-600" /> Manage Users</h2>
              <div className="relative w-full max-w-xs">
                <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
                <input
                  value={searchUsers}
                  onChange={(e) => setSearchUsers(e.target.value)}
                  placeholder="Search users..."
                  className="w-full pl-9 pr-3 py-2 text-sm bg-white border border-slate-200 rounded-lg outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400"
                />
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] text-left">
                <thead className="bg-slate-50 text-slate-500 text-xs uppercase font-bold">
                  <tr>
                    <th className="p-4">Identity</th>
                    <th className="p-4">Role</th>
                    <th className="p-4">Status</th>
                    <th className="p-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm">
                  {loading ? (
                    <tr><td colSpan="4" className="p-8 text-center text-slate-400">Loading users...</td></tr>
                  ) : filteredUsers.map((u) => (
                    <tr key={u._id} className="hover:bg-slate-50">
                      <td className="p-4">
                        <p className="font-semibold text-slate-800">{u.fullName}</p>
                        <p className="text-xs text-slate-500">{u.email}</p>
                      </td>
                      <td className="p-4">
                        <span className="px-2.5 py-1 rounded-md text-xs font-bold border bg-white text-slate-700 border-slate-200">{u.role}</span>
                      </td>
                      <td className="p-4"><span className="text-green-700 text-xs font-bold">Active</span></td>
                      <td className="p-4 text-right">
                        {u.role !== 'Admin' && (
                          <button onClick={() => handleDeleteUser(u._id)} className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg">
                            <Trash2 size={16} />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
              <h3 className="font-bold text-slate-800 flex items-center gap-2 mb-4"><Cpu size={18} className="text-blue-600" /> Runtime Status</h3>
              <StatusRow label="Timezone" value={systemStatus?.timezone || '-'} />
              <StatusRow label="Hair Model Loaded" value={String(!!systemStatus?.hairModelLoaded)} />
              <StatusRow label="Backend" value={systemStatus?.hairModelBackend || '-'} />
              <StatusRow label="Strict Validation" value={String(!!systemStatus?.strictValidation)} />
              <StatusRow label="Fallback Enabled" value={String(!!systemStatus?.allowHairFallback)} />
              {systemStatus?.hairModelError && (
                <div className="mt-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                  {systemStatus.hairModelError}
                </div>
              )}
            </div>

            <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
              <h3 className="font-bold text-slate-800 flex items-center gap-2 mb-4"><Activity size={18} className="text-blue-600" /> Analysis Metrics</h3>
              <StatusRow label="Total Analyses" value={stats.totalAnalyses ?? 0} />
              <StatusRow label="Hair-Only Records" value={stats.hairOnly ?? 0} />
              <StatusRow label="Flagged Records" value={stats.flagged ?? 0} />
              <StatusRow label="Admins" value={stats.admins ?? 0} />
              <StatusRow label="Students" value={stats.students ?? 0} />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between gap-3">
            <h2 className="font-bold text-slate-800 flex items-center gap-2"><Scissors size={18} className="text-blue-600" /> Analysis Records</h2>
            <div className="relative w-full max-w-xs">
              <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
              <input
                value={searchAnalyses}
                onChange={(e) => setSearchAnalyses(e.target.value)}
                placeholder="Search analyses..."
                className="w-full pl-9 pr-3 py-2 text-sm bg-white border border-slate-200 rounded-lg outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400"
              />
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-left">
              <thead className="bg-slate-50 text-slate-500 text-xs uppercase font-bold">
                <tr>
                  <th className="p-4">Date</th>
                  <th className="p-4">Diagnosis</th>
                  <th className="p-4">Result</th>
                  <th className="p-4">Confidence</th>
                  <th className="p-4">Status</th>
                  <th className="p-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {loading ? (
                  <tr><td colSpan="6" className="p-8 text-center text-slate-400">Loading analyses...</td></tr>
                ) : filteredAnalyses.length === 0 ? (
                  <tr><td colSpan="6" className="p-8 text-center text-slate-400">No analysis records found.</td></tr>
                ) : filteredAnalyses.map((r) => (
                  <tr key={r._id} className="hover:bg-slate-50">
                    <td className="p-4 text-slate-600">{r.date || '-'}</td>
                    <td className="p-4 font-semibold text-slate-800">{r.diagnosis || '-'}</td>
                    <td className="p-4 text-slate-600">{r.result || '-'}</td>
                    <td className="p-4 text-slate-600">{r.confidence || '-'}</td>
                    <td className="p-4">
                      <span className="px-2.5 py-1 rounded-md text-xs font-bold border bg-blue-50 text-blue-700 border-blue-100">
                        {r.status || '-'}
                      </span>
                    </td>
                    <td className="p-4 text-right">
                      <button onClick={() => handleDeleteAnalysis(r._id)} className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg">
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

const StatCard = ({ icon, label, value }) => (
  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
    <div className="p-3 bg-blue-50 text-blue-600 rounded-lg">{icon}</div>
    <div>
      <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">{label}</p>
      <h4 className="text-2xl font-bold text-slate-800">{value}</h4>
    </div>
  </div>
);

const StatusRow = ({ label, value }) => (
  <div className="flex items-center justify-between py-1.5">
    <span className="text-sm text-slate-500">{label}</span>
    <span className="text-sm font-semibold text-slate-800">{String(value)}</span>
  </div>
);

export default AdminPanel;
