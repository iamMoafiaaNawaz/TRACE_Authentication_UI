import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { UploadCloud, Activity, Clock, AlertCircle, ArrowRight, FileCheck } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const Dashboard = () => {
  const [userName, setUserName] = useState('Doctor');
  const [stats, setStats] = useState({ totalScans: 0, completed: 0, pending: 0, issues: 0 });
  const [recentScans, setRecentScans] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUserName(parsedUser.fullName); 
      } catch (e) { console.error(e); }
    }
    setTimeout(() => setIsLoading(false), 500);
  }, []);

  return (
    <DashboardLayout>
      {/* FIX 1: 'w-full' aur 'overflow-hidden' 
         Yeh ensure karta hai ke koi bhi cheez parent se bahar na nikle 
      */}
      <div className="animate-fade-in w-full max-w-full space-y-6 md:space-y-8 overflow-hidden">
      
        {/* 1. HEADER */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 w-full">
          <div className="min-w-0"> {/* FIX 2: Text truncate karne ke liye */}
            <h1 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight truncate">Overview</h1>
            <p className="text-slate-500 mt-1 text-sm truncate">
              Welcome back, <span className="font-semibold text-blue-600">{userName}</span>.
            </p>
          </div>
          
          <div className="hidden md:block shrink-0">
            <div className="flex items-center gap-2 text-slate-500 text-sm font-medium bg-white px-3 py-1 rounded-full border border-slate-200 shadow-sm">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
              </span>
              System Online
            </div>
          </div>
        </div>

        {/* 2. STATS CARDS */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
          <StatCard title="Total Scans" value={isLoading ? "-" : stats.totalScans} icon={<Activity size={20} />} />
          <StatCard title="Completed" value={isLoading ? "-" : stats.completed} icon={<FileCheck size={20} />} />
          <StatCard title="Pending" value={isLoading ? "-" : stats.pending} icon={<Clock size={20} />} />
          <StatCard title="Flagged Issues" value={isLoading ? "-" : stats.issues} icon={<AlertCircle size={20} />} />
        </div>

        {/* 3. MAIN ACTION AREA */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full">
          
          {/* Blue Card */}
          <div className="lg:col-span-2 bg-blue-600 rounded-2xl p-6 text-white shadow-lg shadow-blue-200 relative overflow-hidden flex flex-col justify-between gap-6">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
            
            <div className="relative z-10">
              <h2 className="text-xl font-bold mb-2">Start New Analysis</h2>
              <p className="text-blue-100 text-sm max-w-md leading-relaxed">
                Upload dermoscopic images for AI-powered classification.
              </p>
            </div>
            
            <Link 
              to="/dashboard/upload" 
              className="relative z-10 w-full sm:w-auto bg-white text-blue-600 px-6 py-3 rounded-xl font-bold text-sm hover:bg-blue-50 transition-colors flex items-center justify-center gap-2 shadow-sm text-center"
            >
              <UploadCloud size={18} />
              <span>Upload Image</span>
            </Link>
          </div>

          {/* Shortcut Card */}
          <div className="lg:col-span-1 bg-white border border-slate-200 rounded-2xl p-6 flex flex-col justify-center items-start shadow-sm h-full w-full">
            <div className="bg-slate-50 p-3 rounded-xl mb-4 text-slate-400">
              <FileCheck size={24} />
            </div>
            <h2 className="text-lg font-bold text-slate-800 mb-1">Recent Results</h2>
            <p className="text-slate-500 text-sm mb-6">Check your past diagnosis history.</p>
            <Link to="/dashboard/history" className="text-blue-600 font-semibold text-sm flex items-center gap-1 hover:gap-2 transition-all">
              View History <ArrowRight size={16} />
            </Link>
          </div>
        </div>

        {/* 4. RECENT ACTIVITY TABLE */}
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm w-full">
          <div className="px-6 py-5 border-b border-slate-100">
            <h3 className="text-base font-bold text-slate-800">Recent Activity</h3>
          </div>
          
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left min-w-[600px]"> 
              <thead className="bg-slate-50 text-slate-500 text-xs font-bold uppercase tracking-wider">
                <tr>
                  <th className="px-6 py-4">Date</th>
                  <th className="px-6 py-4">Status</th>
                  <th className="px-6 py-4">Result</th>
                  <th className="px-6 py-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {recentScans.length > 0 ? (
                  recentScans.map((item) => (
                    <tr key={item.id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4 text-sm text-slate-600">{item.date}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="4" className="px-6 py-12 text-center text-slate-400 text-sm">
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

// --- FIX 3: StatCard mein 'min-w-0' aur 'flex-1' lagaya ---
const StatCard = ({ title, value, icon }) => (
  <div className="bg-white p-5 rounded-2xl border border-slate-200 flex items-center gap-4 shadow-sm w-full overflow-hidden">
    <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center text-slate-500 border border-slate-100 shrink-0">
      {icon}
    </div>
    
    {/* min-w-0 bohot zaroori hai text overflow rokne ke liye */}
    <div className="flex-1 min-w-0">
      <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider truncate">
        {title}
      </p>
      <h3 className="text-2xl font-bold text-slate-800 mt-0.5 truncate">
        {value}
      </h3>
    </div>
  </div>
);

export default Dashboard;