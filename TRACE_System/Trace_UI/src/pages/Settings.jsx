import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Lock, Save, CheckCircle, AlertCircle, Eye, EyeOff } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const Settings = () => {
  const [formData, setFormData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  
  const [userEmail, setUserEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  
  // Toggles for visibility
  const [showNewPass, setShowNewPass] = useState(false);
  const [showConfirmPass, setShowConfirmPass] = useState(false);

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const parsed = JSON.parse(storedUser);
        if (parsed.email) setUserEmail(parsed.email);
      } catch (e) {
        console.error("Error loading user email", e);
      }
    }
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    if (message.text) setMessage({ type: '', text: '' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage({ type: '', text: '' });

    // 1. Validation: Match Check
    if (formData.newPassword !== formData.confirmPassword) {
      setMessage({ type: 'error', text: "New passwords do not match!" });
      setLoading(false);
      return;
    }

    // 2. Validation: Length Check (UPDATED TO 8)
    if (formData.newPassword.length < 8) {
      setMessage({ type: 'error', text: "Password must be at least 8 characters long." });
      setLoading(false);
      return;
    }

    try {
      // 3. API CALL
      // Note: Make sure Backend URL is correct
      const response = await axios.put('http://localhost:5000/api/auth/update-password', {
        email: userEmail,
        currentPassword: formData.currentPassword,
        newPassword: formData.newPassword
      });

      setMessage({ type: 'success', text: response.data.message || "Password updated successfully!" });
      setFormData({ currentPassword: '', newPassword: '', confirmPassword: '' });

    } catch (err) {
      console.error("Update Error:", err);
      const errorText = err.response?.data?.error || "Failed to update password. Check current password.";
      setMessage({ type: 'error', text: errorText });
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="max-w-3xl mx-auto font-sans mt-8 p-4">
        
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-slate-800">Account Settings</h1>
          <p className="text-slate-500 text-sm">Manage your security preferences.</p>
        </div>

        <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center gap-3">
            <div className="p-2 bg-blue-100 text-[#1E90FF] rounded-lg"><Lock size={20} /></div>
            <div>
              <h3 className="font-bold text-slate-800">Change Password</h3>
              <p className="text-xs text-slate-500">Ensure your account uses a strong password.</p>
            </div>
          </div>

          <div className="p-8">
            {message.text && (
              <div className={`mb-6 p-4 rounded-xl flex items-center gap-3 text-sm font-medium ${
                message.type === 'success' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'
              }`}>
                {message.type === 'success' ? <CheckCircle size={18} /> : <AlertCircle size={18} />}
                {message.text}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              
              {/* Current Password */}
              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Current Password</label>
                <input 
                  type="password" 
                  name="currentPassword"
                  value={formData.currentPassword}
                  onChange={handleChange}
                  placeholder="Enter current password"
                  className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:border-[#1E90FF] focus:ring-2 focus:ring-blue-100 outline-none"
                  required
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* New Password (With Eye Icon) */}
                <div>
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">New Password</label>
                  <div className="relative">
                    <input 
                      type={showNewPass ? "text" : "password"} 
                      name="newPassword"
                      value={formData.newPassword}
                      onChange={handleChange}
                      placeholder="Min 8 chars"
                      className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:border-[#1E90FF] focus:ring-2 focus:ring-blue-100 outline-none"
                      required
                    />
                    <button type="button" onClick={() => setShowNewPass(!showNewPass)} className="absolute right-3 top-3 text-slate-400 hover:text-[#1E90FF]">
                      {showNewPass ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                </div>

                {/* Confirm Password (NEW: With Eye Icon) */}
                <div>
                  <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Confirm New Password</label>
                  <div className="relative">
                    <input 
                      type={showConfirmPass ? "text" : "password"} 
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      placeholder="Repeat new password"
                      className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:border-[#1E90FF] focus:ring-2 focus:ring-blue-100 outline-none"
                      required
                    />
                    <button type="button" onClick={() => setShowConfirmPass(!showConfirmPass)} className="absolute right-3 top-3 text-slate-400 hover:text-[#1E90FF]">
                      {showConfirmPass ? <EyeOff size={18} /> : <Eye size={18} />}
                    </button>
                  </div>
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <button 
                  type="submit" 
                  disabled={loading}
                  className={`flex items-center gap-2 px-6 py-3 bg-[#1E90FF] text-white font-bold rounded-xl shadow-lg shadow-blue-200 hover:bg-blue-600 transition-all ${loading ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                  {loading ? 'Updating...' : <><Save size={18} /> Update Password</>}
                </button>
              </div>

            </form>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default Settings;