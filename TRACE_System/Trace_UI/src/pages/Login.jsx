import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, Eye, EyeOff, ArrowRight, ArrowLeft } from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';
import axios from 'axios';

const Login = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [formData, setFormData] = useState({ 
    email: '', 
    password: '' 
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // --- BACKEND CALL ---
      const response = await axios.post('http://127.0.0.1:5000/api/auth/login', {
        email: formData.email,
        password: formData.password
      });

      console.log("Login Success:", response.data);
      
      // 1. Token Save
      localStorage.setItem('token', response.data.token);
      
      // 2. User Info Save
      localStorage.setItem('user', JSON.stringify(response.data.user));

      setLoading(false);
      
      // 3. Dashboard Redirect
      navigate('/dashboard'); 

    } catch (err) {
      console.error("Login Failed:", err);
      setError(err.response?.data?.error || "Invalid Credentials or Server Offline");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-50 relative overflow-hidden font-sans">
      
      {/* Decorative Background */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-blue-300/20 rounded-full blur-[100px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-[#1E90FF]/20 rounded-full blur-[100px]" />

      {/* Back to Home Button */}
      <Link to="/" className="absolute top-6 left-6 flex items-center gap-2 text-slate-500 hover:text-[#1E90FF] transition-colors font-bold text-sm z-20">
        <ArrowLeft size={20} /> Back to Home
      </Link>

      <div className="relative z-10 w-full max-w-md bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl shadow-blue-100 border border-white/60 p-8 md:p-10">
        
        <div className="text-center mb-8">
          <div className="inline-flex justify-center items-center w-16 h-16 bg-white rounded-2xl shadow-sm mb-4 p-3">
            <img src={traceLogo} alt="TRACE" className="w-full h-full object-contain" />
          </div>
          <h2 className="text-2xl font-bold text-slate-800 tracking-tight">Welcome Back</h2>
          <p className="text-slate-500 text-sm mt-1">Sign in to TRACE AI System</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-5">
          
          {/* Email Field */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2 ml-1">Email Address</label>
            <div className="relative group">
              <Mail className="absolute left-4 top-3.5 text-slate-400 group-focus-within:text-[#1E90FF] transition-colors" size={20} />
              <input 
                type="email" 
                required
                className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-slate-50 border border-slate-200 focus:bg-white focus:border-[#1E90FF] focus:ring-4 focus:ring-blue-500/10 outline-none transition-all font-medium text-slate-700"
                placeholder="doctor@hospital.com"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
              />
            </div>
          </div>

          {/* Password Field */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2 ml-1">Password</label>
            <div className="relative group">
              <Lock className="absolute left-4 top-3.5 text-slate-400 group-focus-within:text-[#1E90FF] transition-colors" size={20} />
              <input 
                type={showPassword ? "text" : "password"} 
                required
                className="w-full pl-12 pr-12 py-3.5 rounded-xl bg-slate-50 border border-slate-200 focus:bg-white focus:border-[#1E90FF] focus:ring-4 focus:ring-blue-500/10 outline-none transition-all font-medium text-slate-700"
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
              />
              <button 
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-3.5 text-slate-400 hover:text-[#1E90FF] transition-colors"
              >
                {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <p className="text-sm text-red-600 bg-red-50 p-3 rounded-lg text-center border border-red-100">
              {error}
            </p>
          )}

          <div className="flex items-center justify-between text-sm mt-1">
            <label className="flex items-center gap-2 cursor-pointer text-slate-600 hover:text-slate-800">
              <input type="checkbox" className="w-4 h-4 accent-[#1E90FF] rounded" />
              Remember me
            </label>
            
            {/* --- FORCE NAVIGATION FIX (Clickable Span) --- */}
            <span 
              onClick={() => navigate('/forgot-password')} 
              className="text-[#1E90FF] font-semibold hover:underline cursor-pointer"
            >
              Forgot Password?
            </span>

          </div>

          <button 
            type="submit"
            disabled={loading}
            className="mt-4 w-full bg-[#1E90FF] hover:bg-blue-600 active:scale-[0.98] text-white font-bold py-4 rounded-xl shadow-lg shadow-blue-500/30 transition-all flex items-center justify-center gap-2 disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                Login to Account <ArrowRight size={20} />
              </>
            )}
          </button>
        </form>

        <p className="text-center mt-8 text-slate-500 text-sm">
          Don't have an account?{' '}
          <Link to="/signup" className="text-[#1E90FF] font-bold hover:underline">
            Create Account
          </Link>
        </p>
      </div>
    </div>
  );
};

export default Login;