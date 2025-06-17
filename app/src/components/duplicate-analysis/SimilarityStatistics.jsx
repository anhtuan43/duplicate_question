import React from 'react';

const SimilarityStatistics = ({ stats }) => (
  <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
    <h3 className="text-xl font-semibold text-gray-800 mb-4">
      THỐNG KÊ SIMILARITY
    </h3>
    <div className="space-y-6">
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-red-50 to-red-100 rounded-lg">
        <span className="font-medium text-gray-700">Max similarity:</span>
        <span className="text-xl font-bold text-red-600">{stats.max.toFixed(4)}</span>
      </div>
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg">
        <span className="font-medium text-gray-700">Avg similarity:</span>
        <span className="text-xl font-bold text-blue-600">{stats.avg.toFixed(4)}</span>
      </div>
      <div className="flex items-center justify-between p-4 bg-gradient-to-r from-green-50 to-green-100 rounded-lg">
        <span className="font-medium text-gray-700">Min similarity:</span>
        <span className="text-xl font-bold text-green-600">{stats.min.toFixed(4)}</span>
      </div>
      <div className="mt-6">
        <div className="h-4 bg-gradient-to-r from-green-400 via-blue-400 to-red-400 rounded-full relative">
          <div className="absolute w-3 h-3 bg-white border-2 border-green-600 rounded-full top-0.5 transform -translate-x-1/2" style={{left: `${(stats.min / 1.0) * 100}%`}}></div>
          <div className="absolute w-3 h-3 bg-white border-2 border-blue-600 rounded-full top-0.5 transform -translate-x-1/2" style={{left: `${(stats.avg / 1.0) * 100}%`}}></div>
          <div className="absolute w-3 h-3 bg-white border-2 border-red-600 rounded-full top-0.5 transform -translate-x-1/2" style={{left: `${(stats.max / 1.0) * 100}%`}}></div>
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-2">
          <span>0.0</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
      </div>
    </div>
  </div>
);

export default SimilarityStatistics;