import React, { useState, useEffect } from 'react';
import { User, Mail, ShieldCheck, Zap } from 'lucide-react';
import DashboardLayout from '../components/DashboardLayout';

const Profile = () => {
  const [user, setUser] = useState({
    fullName: 'Loading...',
    email: 'Loading...',
    role: 'Loading...'
  });

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch (error) {
        console.error("Profile Load Error", error);
      }
    }
  }, []);

  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto font-sans mt-8 p-4">

        {/* Main profile container */}
        <div className="bg-white rounded-3xl border border-slate-200 overflow-hidden 
                        flex flex-col md:flex-row w-full">

          {/* Left Section */}
          <div className="md:w-1/3 bg-gradient-to-br from-blue-400 to-blue-600 p-10 
                          flex flex-col items-center text-white justify-center">

            <div className="w-28 h-28 rounded-full bg-white text-blue-600 flex items-center 
                            justify-center text-5xl font-bold shadow-sm">
              {user.fullName ? user.fullName.charAt(0).toUpperCase() : 'U'}
            </div>

            <h2 className="text-2xl font-semibold mt-6">{user.fullName}</h2>

            <span className="mt-2 px-4 py-1 bg-white/15 backdrop-blur-md rounded-full 
                             text-sm border border-white/20">
              {user.role} Account
            </span>
          </div>

          {/* Right Section */}
          <div className="md:w-2/3 p-10 bg-white">

            <h3 className="text-2xl font-semibold text-slate-800 mb-1">
              Profile Details
            </h3>
            <p className="text-slate-500 text-sm mb-8">
              Here is your personal account information.
            </p>

            <div className="space-y-4">

              <ProfileRow
                icon={<User size={20} />}
                label="Full Name"
                value={user.fullName}
              />

              <ProfileRow
                icon={<Mail size={20} />}
                label="Email Address"
                value={user.email}
              />

              <ProfileRow
                icon={<ShieldCheck size={20} />}
                label="Account Role"
                value={user.role}
                capitalize
              />

              <ProfileRow
                icon={<Zap size={20} className="text-green-500" />}
                label="Status"
                value="Active & Verified"
                valueColor="text-green-600 font-medium"
              />
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
};

const ProfileRow = ({
  icon,
  label,
  value,
  capitalize = false,
  valueColor = "text-slate-900"
}) => (
  <div className="flex items-center gap-4 p-4 rounded-2xl border border-slate-200 bg-slate-50/40">

    <div className="w-12 h-12 flex items-center justify-center rounded-xl bg-blue-50 text-blue-600">
      {icon}
    </div>

    <div className="flex-1">
      <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
        {label}
      </p>
      <p className={`text-base font-medium ${valueColor} ${capitalize ? "capitalize" : ""}`}>
        {value}
      </p>
    </div>
  </div>
);

export default Profile;
