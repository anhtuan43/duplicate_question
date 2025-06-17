import React from 'react';
import { FileText, Copy, Users, TrendingUp, AlertTriangle, BarChart3 } from 'lucide-react';
import StatCard from './StatCard';

const OverviewStats = ({ data }) => (
  <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
    <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
      <BarChart3 className="w-6 h-6 mr-2" />
      TỔNG QUAN
    </h2>
    
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
      <StatCard icon={FileText} value={data.totalQuestions} label="Tổng số câu hỏi" colorClass="blue" />
      <StatCard icon={Users} value={data.uniqueQuestions} label="Câu hỏi unique" colorClass="green" />
      <StatCard icon={Copy} value={data.duplicateQuestions} label="Câu hỏi duplicate" colorClass="red" />
      <StatCard icon={TrendingUp} value={`${data.duplicateRate}%`} label="Tỷ lệ duplicate" colorClass="orange" />
      <StatCard icon={AlertTriangle} value={data.duplicatePairs.toLocaleString()} label="Số cặp duplicate" colorClass="purple" />
    </div>
  </div>
);

export default OverviewStats;