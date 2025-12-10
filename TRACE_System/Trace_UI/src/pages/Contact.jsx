import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Mail, Globe, MessageSquare, Send } from 'lucide-react';
import traceLogo from '../assets/trace-logo.png';
import InputField from '../components/InputField';
import PrimaryButton from '../components/PrimaryButton';

const Contact = () => {
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    
    // Simulate sending email
    setTimeout(() => {
      setLoading(false);
      setSubmitted(true);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-700 flex flex-col">
      
      {/* 1. Navigation Bar */}
      <nav className="bg-white/90 backdrop-blur-md sticky top-0 z-50 border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <img src={traceLogo} alt="TRACE" className="h-10 w-auto object-contain" />
            <div className="flex flex-col">
              <span className="text-lg font-bold text-slate-800 leading-none">TRACE</span>
              <span className="text-[10px] font-semibold text-[#1E90FF] uppercase tracking-wide">Support</span>
            </div>
          </div>
          <Link 
            to="/" 
            className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-[#1E90FF] transition-colors"
          >
            <ArrowLeft size={18} /> Back to Home
          </Link>
        </div>
      </nav>

      {/* 2. Main Content Area */}
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="max-w-5xl w-full bg-white rounded-3xl shadow-xl overflow-hidden border border-slate-100 flex flex-col md:flex-row">
          
          {/* LEFT SIDE: General Info (Blue) */}
          <div className="bg-[#1E90FF] p-10 md:w-2/5 text-white flex flex-col justify-between relative overflow-hidden">
            {/* Background Decoration */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-white/10 rounded-full blur-3xl -mr-10 -mt-10"></div>
            
            <div>
              <h1 className="text-3xl font-bold mb-4">Get in Touch</h1>
              <p className="text-blue-100 text-sm leading-relaxed mb-8">
                Have feedback on the AI predictions or facing account issues? 
                Our support team is here to assist you with any inquiries regarding the TRACE platform.
              </p>

              <div className="space-y-6">
                {/* 1. Updated Email */}
                <ContactItem 
                  icon={<Mail className="text-blue-200" />}
                  label="Support Email"
                  value="fypbsai@gmail.com"
                />
                
                {/* 2. Generalized Location (Web Platform) */}
                <ContactItem 
                  icon={<Globe className="text-blue-200" />}
                  label="Platform Access"
                  value="Global / Web-Based"
                />
                
                {/* 3. Generalized Response Time */}
                <ContactItem 
                  icon={<MessageSquare className="text-blue-200" />}
                  label="Response Time"
                  value="Typically within 24-48 hours"
                />
              </div>
            </div>

            <div className="mt-12">
              <p className="text-xs text-blue-200 opacity-80">
                For urgent clinical concerns, please consult a dermatologist directly. TRACE is an assistive AI tool.
              </p>
            </div>
          </div>

          {/* RIGHT SIDE: Inquiry Form */}
          <div className="p-10 md:w-3/5 bg-white">
            
            {submitted ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-10">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
                  <Send className="text-green-600 w-8 h-8" />
                </div>
                <h3 className="text-2xl font-bold text-slate-800">Message Sent!</h3>
                <p className="text-slate-500 mt-2 max-w-xs">
                  Thank you for reaching out. We have received your query and will get back to you at <strong>fypbsai@gmail.com</strong> soon.
                </p>
                <button 
                  onClick={() => setSubmitted(false)}
                  className="mt-6 text-[#1E90FF] font-semibold hover:underline"
                >
                  Send another message
                </button>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-5">
                <h2 className="text-xl font-bold text-slate-800 mb-6">Send us a Message</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <InputField label="Your Name" placeholder="Full Name" type="text" required />
                  <InputField label="Email Address" placeholder="you@example.com" type="email" required />
                </div>

                {/* Subject Dropdown */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">Subject</label>
                  <div className="relative">
                    <select className="w-full px-4 py-3.5 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#1E90FF] appearance-none text-slate-700 cursor-pointer">
                      <option>General Inquiry</option>
                      <option>Report a Technical Issue</option>
                      <option>Feedback on AI Results</option>
                      <option>Account Assistance</option>
                    </select>
                  </div>
                </div>

                {/* Message Textarea */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">Message</label>
                  <textarea 
                    rows="4"
                    required
                    className="w-full p-4 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#1E90FF] text-slate-700 placeholder-slate-400 resize-none"
                    placeholder="How can we help you?"
                  ></textarea>
                </div>

                <PrimaryButton type="submit" className="w-full bg-[#1E90FF]" disabled={loading}>
                  {loading ? 'Sending...' : 'Send Message'}
                </PrimaryButton>
              </form>
            )}

          </div>

        </div>
      </div>

    </div>
  );
};

// Helper Component
const ContactItem = ({ icon, label, value }) => (
  <div className="flex items-start gap-4">
    <div className="mt-1 p-2 bg-white/10 rounded-lg backdrop-blur-sm">
      {icon}
    </div>
    <div>
      <p className="text-xs text-blue-200 font-semibold uppercase tracking-wider">{label}</p>
      <p className="text-white font-medium">{value}</p>
    </div>
  </div>
);

export default Contact;