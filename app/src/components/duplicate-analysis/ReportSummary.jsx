import React from 'react';

const ReportSummary = ({ data }) => (
  <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-lg p-6 text-white">
    <div className="text-center">
      <h3 className="text-xl font-semibold mb-2">Tóm tắt phân tích</h3>
      <p className="text-blue-100">
        Từ {data.totalQuestions} câu hỏi, có {data.duplicateQuestions} câu ({data.duplicateRate}%) 
        bị trùng lặp với {data.duplicatePairs} cặp duplicate được phát hiện.
        Similarity trung bình là {data.similarityStats.avg.toFixed(4)}.
      </p>
    </div>
  </div>
);

export default ReportSummary;