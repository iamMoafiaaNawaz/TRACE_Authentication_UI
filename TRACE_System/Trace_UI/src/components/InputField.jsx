import React from 'react';

const InputField = ({ label, type, placeholder, value, onChange, icon, rightIcon, required }) => {
  return (
    <div className="w-full">
      <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">
        {label}
      </label>
      <div className="relative flex items-center">
        {/* Left Icon */}
        <div className="absolute left-4 text-slate-400">
          {icon}
        </div>

        <input
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          required={required}
          className="w-full pl-12 pr-12 py-3.5 bg-white/70 backdrop-blur-sm border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#1E90FF]/50 focus:border-[#1E90FF] transition-all text-slate-800 placeholder-slate-400 shadow-sm"
        />

        {/* Right Icon (Password Eye) */}
        {rightIcon && (
          <div className="absolute right-4">
            {rightIcon}
          </div>
        )}
      </div>
    </div>
  );
};

export default InputField;