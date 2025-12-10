import React, { useState, useEffect } from 'react';
import { FileText, Download, Calendar, Search, AlertCircle } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';
import axios from 'axios'; // Future API call ke liye

const History = () => {
  // --- STATE MANAGEMENT ---
  // 1. history: Yahan backend ka data aayega (abhi empty hai)
  const [history, setHistory] = useState([]); 
  // 2. loading: Jab tak data nahi aata, ye true rahega
  const [loading, setLoading] = useState(true);
  // 3. searchTerm: Search ke liye
  const [searchTerm, setSearchTerm] = useState('');

  // --- API CALL PLACEHOLDER ---
  useEffect(() => {
    // Yahan aap bad mein Backend API call lagayengi
    // Example: axios.get('/api/history').then(res => setHistory(res.data));
    
    // Filhal hum simulate kar rahe hain ke data load ho gaya (aur empty hai)
    setTimeout(() => {
      setLoading(false);
    }, 1000);
  }, []);

  // Filter Logic (Jab data aayega tab ye kaam karega)
  const filteredHistory = history.filter(item => 
    item.date.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.diagnosis.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <DashboardLayout>
      <div className="w-full max-w-6xl mx-auto font-sans animate-fade-in space-y-6">
        
        {/* --- HEADER SECTION --- */}
        {/* Flexbox: Mobile (Column), Desktop (Row) */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-slate-800 tracking-tight">Analysis History</h1>
            <p className="text-slate-500 text-sm mt-1">Archive of your past dermoscopic scans.</p>
          </div>
          
          {/* Search Input: Mobile (Full Width), Desktop (Fixed Width) */}
          <div className="relative w-full md:w-64">
            <Search className="absolute left-3 top-2.5 text-slate-400" size={18} />
            <input 
              type="text" 
              placeholder="Search history..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white border border-slate-200 rounded-xl text-sm outline-none focus:border-[#1E90FF] focus:ring-2 focus:ring-blue-50 transition-all shadow-sm"
            />
          </div>
        </div>
        
        {/* --- TABLE SECTION --- */}
        <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm w-full">
          
          {/* Responsive Table Trick: 
             'overflow-x-auto' lagane se agar screen choti hogi to table scroll karegi 
             bajaye page ko kharab karne ke.
          */}
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left min-w-[700px]"> 
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 text-xs font-bold uppercase tracking-wider">
                <tr>
                  <th className="px-6 py-4 whitespace-nowrap">Date Scanned</th>
                  <th className="px-6 py-4 whitespace-nowrap">Diagnosis Result</th>
                  <th className="px-6 py-4 whitespace-nowrap">Confidence</th>
                  <th className="px-6 py-4 whitespace-nowrap">Risk Level</th>
                  <th className="px-6 py-4 text-right whitespace-nowrap">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {/* LOADING STATE */}
                {loading ? (
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center text-slate-400">
                      <div className="flex justify-center items-center gap-2">
                        <span className="w-4 h-4 border-2 border-slate-300 border-t-blue-500 rounded-full animate-spin"></span>
                        Loading records...
                      </div>
                    </td>
                  </tr>
                ) : filteredHistory.length === 0 ? (
                  /* EMPTY STATE (No Data) */
                  <tr>
                    <td colSpan="5" className="px-6 py-12 text-center">
                      <div className="flex flex-col items-center justify-center text-slate-400">
                        <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mb-3">
                          <AlertCircle size={24} />
                        </div>
                        <p className="text-sm font-medium text-slate-600">No history found</p>
                        <p className="text-xs mt-1">You haven't performed any analysis yet.</p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  /* DATA MAPPING (Jab Backend se data aayega tab ye chalega) */
                  filteredHistory.map((item) => (
                    <tr key={item.id} className="hover:bg-slate-50 transition-colors">
                      <td className="px-6 py-4 text-sm text-slate-600 flex items-center gap-2 whitespace-nowrap">
                        <Calendar size={16} className="text-slate-400" />
                        {item.date}
                      </td>
                      <td className="px-6 py-4 text-sm font-bold text-slate-800 whitespace-nowrap">
                        {item.diagnosis}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-600 whitespace-nowrap">
                        {item.confidence}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold border ${
                          item.status === 'Suspicious' 
                            ? 'bg-red-50 text-red-700 border-red-100' 
                            : 'bg-green-50 text-green-700 border-green-100'
                        }`}>
                          {item.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right flex justify-end gap-3 whitespace-nowrap">
                        <button className="text-slate-400 hover:text-[#1E90FF] transition-colors p-1 hover:bg-blue-50 rounded-lg" title="View Report">
                          <FileText size={18} />
                        </button>
                        <button className="text-slate-400 hover:text-[#1E90FF] transition-colors p-1 hover:bg-blue-50 rounded-lg" title="Download">
                          <Download size={18} />
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </DashboardLayout>
  );
};

export default History;