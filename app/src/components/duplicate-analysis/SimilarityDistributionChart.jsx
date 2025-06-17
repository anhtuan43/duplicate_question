import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label } from 'recharts';

// --- COMPONENT MỚI: Tooltip tùy chỉnh ---
// Tạo một component riêng để style tooltip, mang lại cảm giác "branded" hơn.
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white/80 backdrop-blur-sm rounded-lg shadow-lg border border-slate-200/50 p-3 text-sm">
        <p className="font-bold text-slate-800 mb-1">{`Khoảng: ${label}`}</p>
        <p className="text-indigo-600 font-semibold">
          {`Số cặp: ${payload[0].value.toLocaleString()}`}
        </p>
      </div>
    );
  }
  return null;
};


const SimilarityDistributionChart = ({ data }) => {
  // Logic đảo ngược dữ liệu vẫn giữ nguyên vì nó rất hiệu quả
  const chartData = data ? data.slice().reverse() : [];

  return (
    // --- CARD CONTAINER ĐƯỢC CẢI TIẾN ---
    <div className="bg-white rounded-xl shadow-lg border border-slate-200/80">
      {/* Header của Card */}
      <div className="p-5 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-100 text-indigo-600 rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8" />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-slate-800">
            Phân Bố Similarity
          </h3>
        </div>
      </div>

      {/* Phần thân card chứa biểu đồ */}
      <div className="p-4 pt-6">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            margin={{ top: 5, right: 10, left: -10, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />

            {/* Trục X được style lại */}
            <XAxis
              dataKey="range"
              angle={-40}
              textAnchor="end"
              height={50}
              interval={0}
              tick={{ fontSize: 12, fill: '#64748b' }} // Màu slate-500
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }} // Màu slate-300
            />

            {/* Trục Y được style lại */}
            <YAxis
              tick={{ fontSize: 12, fill: '#64748b' }}
              tickLine={false}
              axisLine={false}
            >
              <Label
                value="Số cặp"
                angle={-90}
                position="insideLeft"
                style={{ textAnchor: 'middle', fill: '#475569' }} // Màu slate-600
              />
            </YAxis>

            {/* Sử dụng Tooltip tùy chỉnh */}
            <Tooltip
              cursor={{ fill: 'rgba(239, 246, 255, 0.6)' }} // Màu blue-50 với opacity
              content={<CustomTooltip />}
            />

            {/* Bar được style lại */}
            <Bar
              dataKey="pairs"
              className="fill-indigo-500" // Dùng class của Tailwind
              radius={[4, 4, 0, 0]}
              barSize={30} // Điều chỉnh độ rộng của cột
              activeBar={{ className: "fill-indigo-600" }} // Hiệu ứng khi hover
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SimilarityDistributionChart;