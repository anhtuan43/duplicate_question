import React from 'react';

const StatCard = ({ icon: Icon, value, label, colorClass }) => {
  const colorVariants = {
    blue: { bg: 'bg-blue-50', text: 'text-blue-600' },
    green: { bg: 'bg-green-50', text: 'text-green-600' },
    red: { bg: 'bg-red-50', text: 'text-red-600' },
    orange: { bg: 'bg-orange-50', text: 'text-orange-600' },
    purple: { bg: 'bg-purple-50', text: 'text-purple-600' },
  };

  const colors = colorVariants[colorClass] || colorVariants.blue;

  return (
    <div className={`text-center p-4 ${colors.bg} rounded-lg`}>
      <Icon className={`w-8 h-8 ${colors.text} mx-auto mb-2`} />
      <div className={`text-2xl font-bold ${colors.text}`}>{value}</div>
      <div className="text-sm text-gray-600">{label}</div>
    </div>
  );
};

export default StatCard;