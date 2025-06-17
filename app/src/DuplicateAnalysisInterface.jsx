// src/DuplicateAnalysisInterface.js
import React, { useState, useMemo, useEffect, useCallback } from 'react';
// Import các component con (không thay đổi)
import ReportHeader from './components/duplicate-analysis/ReportHeader';
import OverviewStats from './components/duplicate-analysis/OverviewStats';
import SimilarityDistributionChart from './components/duplicate-analysis/SimilarityDistributionChart';
import SimilarityStatistics from './components/duplicate-analysis/SimilarityStatistics';
import TopDuplicatesTable from './components/duplicate-analysis/TopDuplicatesTable';
import ReportSummary from './components/duplicate-analysis/ReportSummary';

// Component giả lập để hiển thị trạng thái loading và error
const LoadingSpinner = () => (
    <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-blue-500"></div>
    </div>
);

const ErrorMessage = ({ message }) => (
    <div className="flex justify-center items-center h-screen bg-red-50">
        <div className="text-center p-8 border border-red-300 rounded-lg bg-white">
            <h2 className="text-2xl font-bold text-red-600 mb-2">Đã xảy ra lỗi</h2>
            <p className="text-gray-700">{message}</p>
        </div>
    </div>
);

const DuplicateAnalysisInterface = () => {
    // State cho API và dữ liệu
    const [reportData, setReportData] = useState(null);
    const [isLoading, setIsLoading] = useState(true); // Bắt đầu ở trạng thái loading
    const [error, setError] = useState(null);


    const initialThreshold = useMemo(() => {
        const stored = localStorage.getItem('threshold');
        return stored ? parseFloat(stored) : 0.85;
    }, []);

    const [threshold, setThreshold] = useState(initialThreshold);

    useEffect(() => {
        localStorage.setItem('threshold', threshold.toString());
    }, [threshold]);

    // --- LOGIC MỚI: Hàm fetch được tái sử dụng ---
    const fetchAnalysisReport = useCallback(async (currentThreshold) => {
        setIsLoading(true);
        setError(null);
        try {
            // Thay đổi endpoint API
            const response = await fetch('http://127.0.0.1:8000/api/analyze/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    threshold: currentThreshold, // Chỉ gửi threshold
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Lỗi từ server: ${response.status}`);
            }

            const data = await response.json();
            console
            setReportData(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, []); // useCallback để hàm này không bị tạo lại mỗi lần render

    // --- LOGIC MỚI: Tự động fetch khi component được tải lần đầu ---
    useEffect(() => {
        // Gọi fetch với ngưỡng mặc định khi component mount
        fetchAnalysisReport(threshold);
    }, [fetchAnalysisReport, threshold]); // Sửa dependency để bao gồm threshold

    // --- LOGIC MỚI: Hàm xử lý khi nhấn nút ---
    const handleRerunAnalysis = () => {
        // Gọi fetch với giá trị threshold hiện tại từ state
        fetchAnalysisReport(threshold);
    };

    // useMemo để tính toán stats được lọc phía client
    const filteredClientSideStats = useMemo(() => {
        if (!reportData || !reportData.topDuplicates) return null;

        // Lọc duplicates theo threshold hiện tại
        const filteredDuplicates = reportData.topDuplicates.filter(
            item => item.similarity >= threshold
        );

        // Tính toán lại stats
        return {
            totalDuplicates: filteredDuplicates.length,
            averageSimilarity: filteredDuplicates.length > 0
                ? filteredDuplicates.reduce((sum, item) => sum + item.similarity, 0) / filteredDuplicates.length
                : 0,
            highSimilarityCount: filteredDuplicates.filter(item => item.similarity >= 0.9).length,
        };
    }, [reportData, threshold]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-6">
            <div className="max-w-7xl mx-auto space-y-6">
                <ReportHeader />

                {/* Khung điều khiển, không còn textarea */}
                <div className="bg-white p-6 rounded-xl shadow-md border border-gray-200">
                    <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                        <div className="flex-grow w-full">
                            <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-1">
                                Điều chỉnh Ngưỡng Tương đồng ({threshold.toFixed(2)})
                            </label>
                            <input
                                id="threshold"
                                type="range"
                                min="0.5"
                                max="1.0"
                                step="0.01"
                                value={threshold}
                                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>
                </div>

                {/* Phần hiển thị kết quả */}
                {isLoading && <LoadingSpinner />}
                {error && !isLoading && <ErrorMessage message={error} />}
                {reportData && !isLoading && !error && (
                    <>
                        <OverviewStats data={reportData} />
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <SimilarityDistributionChart data={reportData.similarityDistribution} />
                            <SimilarityStatistics stats={reportData.similarityStats} />
                        </div>
                        <TopDuplicatesTable data={reportData.topDuplicates} currentThreshold={threshold} />
                        <ReportSummary data={reportData} />
                    </>
                )}
            </div>
        </div>
    );
};

export default DuplicateAnalysisInterface;