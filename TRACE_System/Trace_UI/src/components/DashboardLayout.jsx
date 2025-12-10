import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, 
  UploadCloud, 
  History, 
  LogOut, 
  Menu, 
  X, 
  User, 
  Settings, 
  ChevronDown,
  ShieldCheck // Sirf ye icon chahiye Admin ke liye
} from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';

const DashboardLayout = ({ children }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const dropdownRef = useRef(null);

  const [user, setUser] = useState({ fullName: 'User', role: 'Guest' });

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (error) {
        console.error("Error parsing user data", error);
      }
    }
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsProfileOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = () => {
    localStorage.clear(); 
    window.location.href = '/login';
  };

  // --- GENERAL LINKS (Sabke liye) ---
  const generalItems = [
    { name: 'Overview', icon: <LayoutDashboard size={20} />, path: '/dashboard' },
    { name: 'New Analysis', icon: <UploadCloud size={20} />, path: '/dashboard/upload' },
    { name: 'History', icon: <History size={20} />, path: '/dashboard/history' },
  ];

  // Helper for Active Link Styling
  const NavLink = ({ item, isActiveOverride }) => {
    const isActive = isActiveOverride || location.pathname === item.path;
    return (
      <Link
        to={item.path}
        onClick={() => setIsMobileMenuOpen(false)}
        className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-sm font-medium ${
          isActive 
            ? 'bg-slate-800 text-white shadow-md'
            : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
        }`}
      >
        {item.icon}
        {item.name}
      </Link>
    );
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC] flex font-sans text-slate-800 overflow-hidden">
      
      {/* SIDEBAR */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-slate-200 transform transition-transform duration-300 ease-in-out md:translate-x-0 ${isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        
        <div className="h-20 flex items-center gap-3 px-6 border-b border-slate-100">
          <img src={traceLogo} alt="TRACE" className="h-8 w-auto" />
          <span className="font-bold text-slate-800 text-lg tracking-tight">TRACE AI</span>
          <button className="md:hidden ml-auto text-slate-500" onClick={() => setIsMobileMenuOpen(false)}>
            <X size={24} />
          </button>
        </div>

        <div className="p-4 space-y-8 overflow-y-auto h-[calc(100vh-80px)]">
          
          {/* 1. Main Menu */}
          <div>
            <p className="px-4 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Main Menu</p>
            <div className="space-y-1">
              {generalItems.map((item) => (
                <NavLink key={item.name} item={item} />
              ))}
            </div>
          </div>

          {/* 2. ADMIN PANEL (SINGLE MODULE) */}
          {/* Sirf Admin ko dikhega, aur sirf 1 button hoga */}
          {user?.role === 'Admin' && (
            <div>
              <p className="px-4 text-xs font-bold text-blue-600 uppercase tracking-wider mb-2 flex items-center gap-2">
                <ShieldCheck size={12} /> Administration
              </p>
              <div className="space-y-1">
                {/* SINGLE LINK: Admin Panel */}
                <Link
                  to="/dashboard/admin"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all text-sm font-medium ${
                    location.pathname === '/dashboard/admin'
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-200' // Alag color taaki highlight ho
                      : 'text-slate-600 hover:bg-slate-100 hover:text-blue-600'
                  }`}
                >
                  <ShieldCheck size={20} />
                  Admin Control Panel
                </Link>
              </div>
            </div>
          )}

        </div>

        <div className="absolute bottom-0 w-full p-4 border-t border-slate-100 bg-white">
          <button 
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-4 py-3 text-red-600 hover:bg-red-50 rounded-lg transition-all text-sm font-medium"
          >
            <LogOut size={20} />
            Sign Out
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT WRAPPER */}
      <div className="flex-1 md:ml-64 flex flex-col min-h-screen w-full max-w-full overflow-x-hidden relative">
        
        {/* HEADER */}
        <header className="h-20 bg-white sticky top-0 z-30 border-b border-slate-200 px-4 md:px-6 flex items-center justify-between shadow-sm w-full">
          <button className="md:hidden text-slate-500 p-2 hover:bg-slate-100 rounded-lg" onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
            {isMobileMenuOpen ? <X /> : <Menu />}
          </button>

          <div className="flex-1"></div>

          {/* Profile Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button 
              onClick={() => setIsProfileOpen(!isProfileOpen)}
              className="flex items-center gap-3 hover:bg-slate-50 p-1.5 pr-3 rounded-full border border-transparent hover:border-slate-200 transition-all outline-none"
            >
              <div className="w-9 h-9 bg-slate-800 rounded-full flex items-center justify-center text-white font-bold text-sm shadow-md shrink-0">
                {user.fullName ? user.fullName.charAt(0).toUpperCase() : 'U'}
              </div>
              <div className="text-left hidden sm:block">
                <p className="text-sm font-bold text-slate-700 leading-tight">{user.fullName}</p>
                <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wide">{user.role}</p>
              </div>
              <ChevronDown size={14} className="text-slate-400" />
            </button>

            {isProfileOpen && (
              <div className="absolute right-0 mt-3 w-56 bg-white rounded-xl shadow-xl border border-slate-100 overflow-hidden animate-fade-in z-50">
                <div className="p-2">
                  <Link to="/dashboard/profile" onClick={() => setIsProfileOpen(false)} className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-slate-600 hover:bg-slate-50 rounded-lg">
                    <User size={16} /> My Profile
                  </Link>
                  <Link to="/dashboard/settings" onClick={() => setIsProfileOpen(false)} className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-slate-600 hover:bg-slate-50 rounded-lg">
                    <Settings size={16} /> Security Settings
                  </Link>
                  <div className="h-px bg-slate-100 my-1"></div>
                  <button onClick={handleLogout} className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 rounded-lg font-medium">
                    <LogOut size={16} /> Sign Out
                  </button>
                </div>
              </div>
            )}
          </div>
        </header>

        {/* PAGE CONTENT */}
        <main className="p-4 md:p-6 w-full max-w-full">
          {children}
        </main>
      </div>
      
      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div className="fixed inset-0 bg-slate-900/20 z-40 md:hidden backdrop-blur-sm" onClick={() => setIsMobileMenuOpen(false)} />
      )}
    </div>
  );
};

export default DashboardLayout;