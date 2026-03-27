import React from 'react';
import { motion } from 'framer-motion';

const PrimaryButton = ({ children, onClick, type = 'button', className = '', disabled = false }) => {
  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`py-3 px-6 rounded-full font-bold text-lg shadow-lg transition-all duration-300 flex items-center justify-center
        ${disabled 
          ? 'bg-blue-300 cursor-not-allowed opacity-70 text-white' 
          : 'bg-[#1E90FF] hover:bg-blue-600 text-white'
        }
        ${className}`}
    >
      {children}
    </motion.button>
  );
};

export default PrimaryButton;