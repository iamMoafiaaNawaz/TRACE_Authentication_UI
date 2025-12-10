import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// --- 1. Public Pages ---
import Welcome from './pages/Welcome';
import Login from './pages/Login';
import Signup from './pages/Signup';
import About from './pages/About';
import Terms from './pages/Terms';
import Contact from './pages/Contact';
import ForgotPassword from './pages/ForgotPassword';

import AdminPanel from './pages/AdminPanel';
// --- 2. Dashboard Pages ---
import Dashboard from './pages/Dashboard';
import NewAnalysis from './pages/NewAnalysis';
import History from './pages/History';
import Profile from './pages/Profile';
import Settings from './pages/Settings'; // <--- YEH IMPORT ZAROORI HAI

function App() {
  return (
    <Router>
      <Routes>
        
        {/* Public Routes */}
        <Route path="/" element={<Welcome />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/about" element={<About />} />
        <Route path="/terms" element={<Terms />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        {/* Dashboard Routes */}
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/dashboard/upload" element={<NewAnalysis />} />
        <Route path="/dashboard/history" element={<History />} />
        <Route path="/dashboard/profile" element={<Profile />} />
        <Route path="/dashboard/admin" element={<AdminPanel />} />
        {/* --- YEH LINE ADD KARNI THI 👇 --- */}
        <Route path="/dashboard/settings" element={<Settings />} />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />

      </Routes>
    </Router>
  );
}

export default App;