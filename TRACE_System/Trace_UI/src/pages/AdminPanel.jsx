import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom'; // <--- Import zaroori hai
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
  RefreshCw,     
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const AdminPanel = () => {
  const [users, setUsers] = useState([]);
  const [stats, setStats] = useState({ totalUsers: 0, students: 0, clinicians: 0, admins: 0 });
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  
  // URL se Hash (#) read karne ke liye hook
  const location = useLocation();

  // --- FETCH DATA ---
  const fetchData = async () => {
    try {
      const [usersRes, statsRes] = await Promise.all([
        axios.get('http://localhost:5000/api/admin/users'),
        axios.get('http://localhost:5000/api/admin/analytics')
      ]);
      setUsers(usersRes.data);
      setStats(statsRes.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching admin data", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // --- SMOOTH SCROLL LOGIC ---
  // Jab bhi URL change ho (click karne par), ye chalega
  useEffect(() => {
    if (location.hash) {
      const id = location.hash.replace('#', '');
      const element = document.getElementById(id);
      if (element) {
        // Thoda sa wait karte hain taaki data load ho jaye, phir scroll ho
        setTimeout(() => {
          element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
      }
    }
  }, [location, loading]); // Loading khatam hone par bhi check karega

  // --- DELETE USER ---
  const handleDelete = async (userId) => {
    if (window.confirm("Are you sure you want to permanently delete this user?")) {
      try {
        await axios.delete(`http://localhost:5000/api/admin/users/${userId}`);
        fetchData(); 
      } catch (error) {
        alert("Failed to delete user");
      }
    }
  };

  const filteredUsers = users.filter(user => 
    user.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <DashboardLayout>
      <div className="max-w-7xl mx-auto font-sans animate-fade-in p-2 space-y-12">
        
        {/* --- HEADER --- */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-slate-200 pb-6">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 tracking-tight">Admin Console</h1>
            <p className="text-slate-500 mt-1">System monitoring, user management, and AI model control.</p>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-bold border border-slate-200">
             <ShieldAlert size={16} className="text-blue-600"/> Administrator Access
          </div>
        </div>

        {/* =========================================
            SECTION 1: VIEW ANALYTICS 
            ID: 'analytics' (scroll-mt-24 header ke neeche chupne se bachata hai)
           ========================================= */}
        <section id="analytics" className="scroll-mt-24">
          <div className="flex items-center gap-2 mb-4">
            <Activity size={20} className="text-blue-600" />
            <h2 className="text-lg font-bold text-slate-700 uppercase tracking-wider">System Analytics</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <StatCard icon={<Users />} label="Total Users" value={stats.totalUsers} />
            <StatCard icon={<ShieldAlert />} label="Admins" value={stats.admins} />
            <StatCard icon={<Stethoscope />} label="Clinicians" value={stats.clinicians} />
            <StatCard icon={<Activity />} label="Students" value={stats.students} />
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* =========================================
              SECTION 2: MANAGE USERS
              ID: 'users'
             ========================================= */}
          <div id="users" className="lg:col-span-2 space-y-4 scroll-mt-24">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Users size={20} className="text-blue-600" />
                <h2 className="text-lg font-bold text-slate-700 uppercase tracking-wider">Manage Users</h2>
              </div>
              
              <div className="relative">
                <Search className="absolute left-3 top-2.5 text-slate-400" size={16} />
                <input 
                  type="text" 
                  placeholder="Search database..." 
                  className="pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-lg text-sm outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-200 transition-all w-full md:w-64 shadow-sm"
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse min-w-[600px]">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 text-xs uppercase font-bold">
                    <tr>
                      <th className="p-4">User Identity</th>
                      <th className="p-4">Role</th>
                      <th className="p-4">Status</th>
                      <th className="p-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-sm">
                    {loading ? (
                      <tr><td colSpan="4" className="p-8 text-center text-slate-400">Loading database...</td></tr>
                    ) : filteredUsers.map((user) => (
                      <tr key={user._id} className="hover:bg-slate-50 transition-colors">
                        <td className="p-4">
                          <p className="font-bold text-slate-800">{user.fullName}</p>
                          <p className="text-xs text-slate-500">{user.email}</p>
                        </td>
                        <td className="p-4">
                          <span className={`px-2.5 py-1 rounded-md text-xs font-bold border ${
                            user.role === 'Admin' ? 'bg-slate-100 text-slate-700 border-slate-200' : 
                            (user.role === 'Doctor' || user.role === 'Clinician') ? 'bg-blue-50 text-blue-700 border-blue-100' :
                            'bg-white text-slate-600 border-slate-200'
                          }`}>
                            {user.role}
                          </span>
                        </td>
                        <td className="p-4">
                           <span className="flex items-center gap-1.5 text-green-600 text-xs font-bold">
                             <span className="w-2 h-2 rounded-full bg-green-500"></span> Active
                           </span>
                        </td>
                        <td className="p-4 text-right">
                          {user.role !== 'Admin' && (
                            <button 
                              onClick={() => handleDelete(user._id)}
                              className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all"
                              title="Delete User"
                            >
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
          </div>

          {/* RIGHT COLUMN */}
          <div className="space-y-8">

            {/* =========================================
                SECTION 3: MANAGE MODEL
                ID: 'model'
               ========================================= */}
            <section id="model" className="scroll-mt-24">
              <div className="flex items-center gap-2 mb-4">
                <Cpu size={20} className="text-blue-600" />
                <h2 className="text-lg font-bold text-slate-700 uppercase tracking-wider">Manage AI Model</h2>
              </div>
              
              <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
                <div className="flex justify-between items-center pb-4 border-b border-slate-100">
                  <div>
                    <p className="text-xs font-bold text-slate-400 uppercase">Current Version</p>
                    <p className="font-bold text-slate-800">TRACE-AI v2.4.1</p>
                  </div>
                  <span className="px-2 py-1 bg-green-50 text-green-700 text-xs font-bold rounded border border-green-200">Stable</span>
                </div>
                
                <div className="space-y-2">
                  <button className="w-full py-2 px-4 bg-slate-50 hover:bg-blue-50 text-slate-700 hover:text-blue-700 font-medium text-sm rounded-lg border border-slate-200 hover:border-blue-200 transition-all flex items-center justify-center gap-2">
                    <RefreshCw size={14} /> Retrain Model
                  </button>
                  <button className="w-full py-2 px-4 bg-slate-50 hover:bg-slate-100 text-slate-700 font-medium text-sm rounded-lg border border-slate-200 transition-all">
                    Update Parameters
                  </button>
                </div>
              </div>
            </section>

            {/* =========================================
                SECTION 4: MONITOR LOGS
                ID: 'logs'
               ========================================= */}
            <section id="logs" className="scroll-mt-24">
              <div className="flex items-center gap-2 mb-4">
                <FileText size={20} className="text-blue-600" />
                <h2 className="text-lg font-bold text-slate-700 uppercase tracking-wider">System Logs</h2>
              </div>
              
              <div className="bg-slate-900 rounded-xl border border-slate-800 shadow-sm overflow-hidden">
                <div className="p-4 space-y-3 max-h-64 overflow-y-auto font-mono text-xs">
                  <LogItem type="success" time="10:42 AM" text="System scan completed" />
                  <LogItem type="info" time="10:38 AM" text="New user registered (ID: 849)" />
                  <LogItem type="warning" time="09:15 AM" text="High latency in image upload" />
                  <LogItem type="info" time="08:00 AM" text="Backup scheduled successfully" />
                  <LogItem type="success" time="07:59 AM" text="Server restart completed" />
                </div>
              </div>
            </section>

          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

// --- HELPER COMPONENTS ---

const StatCard = ({ icon, label, value }) => (
  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4 hover:border-blue-300 transition-colors">
    <div className="p-3 bg-blue-50 text-blue-600 rounded-lg">
      {icon}
    </div>
    <div>
      <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">{label}</p>
      <h4 className="text-2xl font-bold text-slate-800">{value}</h4>
    </div>
  </div>
);

const LogItem = ({ type, time, text }) => {
  let colorClass = "text-slate-300";
  let Icon = CheckCircle;
  
  if (type === 'warning') { colorClass = "text-yellow-400"; Icon = AlertTriangle; }
  if (type === 'success') { colorClass = "text-green-400"; Icon = CheckCircle; }
  if (type === 'info') { colorClass = "text-blue-400"; Icon = FileText; }

  return (
    <div className="flex gap-3 items-start border-l-2 border-slate-700 pl-3 py-1">
      <span className="text-slate-500 whitespace-nowrap">{time}</span>
      <span className={colorClass}>{text}</span>
    </div>
  );
};

export default AdminPanel;