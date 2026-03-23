import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, Key, ArrowLeft, CheckCircle } from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';
import axios from 'axios';

const ForgotPassword = () => {
  const navigate = useNavigate();
  const [step, setStep] = useState(1); // Step 1: Email, Step 2: OTP & New Pass
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const [formData, setFormData] = useState({
    email: '',
    otp: '',
    newPassword: ''
  });

  // --- STEP 1: SEND OTP ---
  const handleSendCode = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await axios.post('http://127.0.0.1:5000/api/auth/forgot-password', {
        email: formData.email
      });
      setLoading(false);
      setStep(2); // Move to next screen
      setSuccessMsg("Verification code sent to your email!");
    } catch (err) {
      setLoading(false);
      setError(err.response?.data?.error || "Email not found");
    }
  };

  // --- STEP 2: RESET PASSWORD ---
  const handleResetPassword = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await axios.post('http://127.0.0.1:5000/api/auth/reset-password', {
        email: formData.email,
        otp: formData.otp,
        newPassword: formData.newPassword
      });
      
      setLoading(false);
      alert("Password Updated Successfully! Please Login."); // Notification
      navigate('/login'); // Redirect to Login

    } catch (err) {
      setLoading(false);
      setError(err.response?.data?.error || "Invalid Code or Error");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-50 relative overflow-hidden font-sans">
      
      {/* Background Shapes */}
      <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-blue-300/20 rounded-full blur-[100px]" />
      <div className="absolute bottom-[-10%] right-[-10%] w-96 h-96 bg-[#1E90FF]/20 rounded-full blur-[100px]" />

      <Link to="/login" className="absolute top-6 left-6 flex items-center gap-2 text-slate-500 hover:text-[#1E90FF] transition-colors font-bold text-sm z-20">
        <ArrowLeft size={20} /> Back to Login
      </Link>

      <div className="relative z-10 w-full max-w-md bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl p-8 animate-fade-in">
        
        <div className="text-center mb-6">
          <div className="inline-flex justify-center items-center w-14 h-14 bg-white rounded-2xl shadow-sm mb-3 p-2">
            <img src={traceLogo} alt="TRACE" className="w-full h-full object-contain" />
          </div>
          <h2 className="text-2xl font-bold text-slate-800">Reset Password</h2>
          <p className="text-slate-500 text-sm">
            {step === 1 ? "Enter your email to receive code" : "Set your new password"}
          </p>
        </div>

        {/* --- STEP 1 FORM (Email) --- */}
        {step === 1 && (
          <form onSubmit={handleSendCode} className="flex flex-col gap-4">
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 ml-1">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-4 top-3.5 text-slate-400" size={20} />
                <input 
                  type="email" 
                  required
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-slate-50 border border-slate-200 focus:border-[#1E90FF] outline-none"
                  placeholder="doctor@gmail.com"
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                />
              </div>
            </div>

            {error && <p className="text-sm text-red-600 bg-red-50 p-3 rounded-lg text-center border border-red-100">{error}</p>}

            <button type="submit" disabled={loading} className="w-full bg-[#1E90FF] hover:bg-blue-600 text-white font-bold py-3.5 rounded-xl shadow-lg transition-all">
              {loading ? "Sending..." : "Send Verification Code"}
            </button>
          </form>
        )}

        {/* --- STEP 2 FORM (OTP + New Password) --- */}
        {step === 2 && (
          <form onSubmit={handleResetPassword} className="flex flex-col gap-4">
            
            {successMsg && (
              <div className="flex items-center gap-2 text-sm text-green-700 bg-green-50 p-3 rounded-lg border border-green-100">
                <CheckCircle size={16} /> {successMsg}
              </div>
            )}

            {/* OTP Input */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 ml-1">Verification Code</label>
              <div className="relative">
                <Key className="absolute left-4 top-3.5 text-slate-400" size={20} />
                <input 
                  type="text" 
                  required
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-slate-50 border border-slate-200 focus:border-[#1E90FF] outline-none tracking-widest font-bold"
                  placeholder="123456"
                  maxLength={6}
                  value={formData.otp}
                  onChange={(e) => setFormData({...formData, otp: e.target.value})}
                />
              </div>
            </div>

            {/* New Password Input */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2 ml-1">New Password</label>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 text-slate-400" size={20} />
                <input 
                  type="text" // User ko dikhane ke liye text rakha hai taaki typo na ho
                  required
                  className="w-full pl-12 pr-4 py-3.5 rounded-xl bg-slate-50 border border-slate-200 focus:border-[#1E90FF] outline-none"
                  placeholder="Enter new password"
                  value={formData.newPassword}
                  onChange={(e) => setFormData({...formData, newPassword: e.target.value})}
                />
              </div>
            </div>

            {error && <p className="text-sm text-red-600 bg-red-50 p-3 rounded-lg text-center border border-red-100">{error}</p>}

            <button type="submit" disabled={loading} className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3.5 rounded-xl shadow-lg transition-all">
              {loading ? "Updating..." : "Update Password"}
            </button>
          </form>
        )}

      </div>
    </div>
  );
};

export default ForgotPassword;