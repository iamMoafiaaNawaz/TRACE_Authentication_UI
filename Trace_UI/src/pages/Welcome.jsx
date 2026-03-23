import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import skinBg from '../assets/skin-bg.jpg'; 
import traceLogo from '../assets/trace-logo.png'; 

const Welcome = () => {
  return (
    // MAIN CONTAINER: Flex Column (Upar se neechay content flow karega)
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-white font-sans">
      
      {/* 1. BACKGROUND IMAGE (Responsive & Fixed) */}
      <div 
        className="absolute inset-0 z-0 opacity-40 pointer-events-none" 
        style={{
          backgroundImage: `url(${skinBg})`,
          backgroundSize: 'cover',      // Image poori screen cover karegi
          backgroundPosition: 'center', // Hamesha center dikhega
          backgroundAttachment: 'fixed' // Scroll karne par image hilegi nahi (Professional Look)
        }}
      />
      
      {/* 2. TOP NAVIGATION */}
      {/* Mobile par padding kam (p-4), Laptop par zyada (p-6) */}
      <nav className="relative z-10 w-full p-4 md:p-6 flex justify-between items-center max-w-7xl mx-auto">
        
        {/* LOGO SECTION */}
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 md:w-14 md:h-14 bg-white rounded-2xl shadow-md border border-slate-100 flex items-center justify-center p-2 overflow-hidden transition-transform hover:scale-105">
            <img 
              src={traceLogo} 
              alt="TRACE" 
              className="w-full h-full object-contain" 
            />
          </div>
          {/* Logo Name: Mobile par chota text, Laptop par bada */}
          <span className="text-lg md:text-xl font-bold text-slate-800 tracking-tight block">TRACE</span>
        </div>

        <Link 
          to="/login" 
          className="bg-white/80 backdrop-blur-sm px-5 py-2 md:px-6 md:py-2.5 rounded-full text-sm md:text-base text-slate-700 font-semibold hover:text-[#1E90FF] border border-slate-200 shadow-sm transition-all hover:shadow-md"
        >
          Login
        </Link>
      </nav>

      {/* 3. MAIN HERO SECTION */}
      {/* Flex-1 ka matlab hai bachi hui saari jagah le lo taaki footer neeche rahe */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-4 py-8 md:py-0 max-w-5xl mx-auto">
        
        {/* Badge */}
        <span className="bg-blue-100/90 backdrop-blur-sm text-[#1E90FF] px-4 py-1.5 md:px-5 md:py-2 rounded-full text-xs md:text-sm font-bold mb-6 md:mb-8 border border-blue-200 tracking-wide uppercase shadow-sm animate-fade-in-up">
          AI Dermatology Assistant
        </span>

        {/* Heading: Responsive Text Sizes */}
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-slate-900 mb-6 tracking-tight leading-tight drop-shadow-sm">
          Detect Early. <br className="hidden sm:block" />
          <span className="text-[#1E90FF]">Treat Confidently.</span>
        </h1>

        {/* Subtitle Card */}
        <div className="bg-white/60 backdrop-blur-md p-4 md:p-6 rounded-3xl border border-white/50 shadow-sm mb-8 md:mb-10 max-w-3xl mx-2">
          <p className="text-base md:text-xl text-slate-700 leading-relaxed font-medium">
            TRACE helps medical students and clinicians analyze dermoscopic images 
            with AI-powered precision for faster, reliable skin cancer detection.
          </p>
        </div>

        {/* Action Buttons: Mobile par Column, Desktop par Row */}
        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto px-4 sm:px-0">
          
          <Link 
            to="/login"
            className="w-full sm:w-auto px-8 md:px-10 py-3.5 md:py-4 bg-[#1E90FF] hover:bg-blue-600 text-white rounded-full font-bold text-base md:text-lg transition-all shadow-lg shadow-blue-200 flex items-center justify-center gap-2 transform hover:-translate-y-1 active:scale-95"
          >
            Start Analysis
            <ArrowRight size={20} />
          </Link>
          
          <Link 
            to="/signup"
            className="w-full sm:w-auto px-8 md:px-10 py-3.5 md:py-4 bg-white/90 hover:bg-white text-slate-700 border border-slate-300 hover:border-[#1E90FF] hover:text-[#1E90FF] rounded-full font-bold text-base md:text-lg transition-all flex items-center justify-center shadow-md hover:shadow-lg active:scale-95"
          >
            Create Account
          </Link>
        </div>

      </main>

      {/* 4. FOOTER */}
      <footer className="relative z-10 py-6 md:py-8 text-center border-t border-slate-200/50 bg-white/30 backdrop-blur-sm px-4">
        
        {/* Links wrap ho jayenge agar screen choti hui */}
        <div className="flex flex-wrap justify-center gap-4 md:gap-6 mb-4 text-xs md:text-sm font-semibold text-slate-600">
          <Link to="/about" className="hover:text-[#1E90FF] transition-colors">About Us</Link>
          <span className="text-slate-300 hidden sm:inline">•</span>
          <Link to="/terms" className="hover:text-[#1E90FF] transition-colors">Terms of Service</Link>
          <span className="text-slate-300 hidden sm:inline">•</span>
          <Link to="/contact" className="hover:text-[#1E90FF] transition-colors">Contact Support</Link>
        </div>

        <p className="text-slate-400 font-medium text-[10px] md:text-xs">
          © 2026 TRACE System. Empowering Healthcare with AI.
        </p>
      </footer>

    </div>
  );
};

export default Welcome;