// src/QuestionDetailPage.js
import React, { useState, useEffect } from 'react';
import { useParams, useLocation, Link } from 'react-router-dom';

// Enhanced Loading Spinner
const LoadingSpinner = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex justify-center items-center">
        <div className="relative">
            <div className="animate-spin rounded-full h-20 w-20 border-4 border-transparent border-t-blue-500 border-r-purple-500"></div>
            <div className="absolute top-2 left-2 animate-spin rounded-full h-16 w-16 border-4 border-transparent border-b-pink-400 border-l-indigo-400 animate-reverse-spin"></div>
            <div className="absolute top-6 left-6 w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full animate-pulse"></div>
        </div>
    </div>
);

// Enhanced Error Message
const ErrorMessage = ({ message }) => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-red-50 flex justify-center items-center p-4">
        <div className="w-full max-w-md">
            <div className="bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-2xl border border-red-200">
                <div className="text-center">
                    <div className="mx-auto w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
                        <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </div>
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Đã xảy ra lỗi</h2>
                    <p className="text-gray-600 leading-relaxed">{message || "Một lỗi không mong muốn đã xảy ra. Vui lòng thử lại sau."}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="mt-6 px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors duration-200 font-medium"
                    >
                        Thử lại
                    </button>
                </div>
            </div>
        </div>
    </div>
);

// Enhanced Similarity Bar
const SimilarityBar = ({ score }) => {
    const percentage = score * 100;
    let colorClass = 'from-emerald-400 to-green-500';
    let bgClass = 'bg-emerald-50';
    let textClass = 'text-emerald-700';

    if (score >= 0.8) {
        colorClass = 'from-red-400 to-rose-500';
        bgClass = 'bg-red-50';
        textClass = 'text-red-700';
    } else if (score >= 0.65) {
        colorClass = 'from-amber-400 to-yellow-500';
        bgClass = 'bg-amber-50';
        textClass = 'text-amber-700';
    }

    return (
        <div className="flex-1">
            <div className="flex justify-between items-center mb-1">
                <span className={`text-xs font-semibold ${textClass}`}>
                    {percentage.toFixed(1)}%
                </span>
            </div>
            <div className={`w-full ${bgClass} rounded-full h-2 overflow-hidden`}>
                <div
                    className={`bg-gradient-to-r ${colorClass} h-2 rounded-full transition-all duration-500 ease-out shadow-sm`}
                    style={{ width: `${percentage}%` }}
                ></div>
            </div>
        </div>
    );
};

function QuestionDetailPage() {
    const { questionId } = useParams();
    const location = useLocation();
    const threshold = location.state?.threshold || 0.85;

    const [details, setDetails] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchDetails = async () => {
            setIsLoading(true);
            setError(null);
            try {
                const response = await fetch(`http://127.0.0.1:8000/api/question-details/${questionId}/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold }),
                });

                if (!response.ok) {
                    const errData = await response.json().catch(() => ({ error: 'Lỗi không xác định từ server.' }));
                    throw new Error(errData.error || `Lỗi ${response.status}: Không thể lấy dữ liệu chi tiết.`);
                }

                const data = await response.json();
                setDetails(data);

            } catch (err) {
                setError(err.message);
            } finally {
                setIsLoading(false);
            }
        };

        fetchDetails();
    }, [questionId, threshold]);

    if (isLoading) return <LoadingSpinner />;
    if (error) return <ErrorMessage message={error} />;
    if (!details) return <ErrorMessage message="Không tìm thấy dữ liệu cho câu hỏi này." />;

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
            <div className="max-w-6xl mx-auto p-4 sm:p-6 lg:p-8">
                {/* Header với Back Button */}
                <div className="mb-8">
                    <Link
                        to="/"
                        className="group inline-flex items-center gap-2 bg-white/80 backdrop-blur-sm px-6 py-3 rounded-xl shadow-lg border border-white/50 text-slate-700 hover:bg-white hover:shadow-xl transition-all duration-300 font-medium"
                    >
                        <svg className="w-5 h-5 transition-transform group-hover:-translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                        Quay lại Báo cáo chính
                    </Link>
                </div>

                {/* Main Question Card */}
                <div className="bg-white/70 backdrop-blur-sm p-8 rounded-2xl shadow-xl border border-white/50 mb-8">
                    <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <div className="flex-1">
                            <h1 className="text-2xl sm:text-3xl font-bold text-gray-800 mb-3">
                                Chi tiết câu hỏi
                                <span className="ml-2 px-3 py-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg text-lg font-semibold">
                                    #{details.main_question.id}
                                </span>
                            </h1>
                            <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-6 rounded-xl border-l-4 border-blue-400">
                                <p className="text-lg text-gray-800 leading-relaxed italic">
                                    "{details.main_question.text}"
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Similar Questions Section */}
                <div className="bg-white rounded-2xl shadow-lg border border-slate-200/80 overflow-hidden">
                    {/* ===== HEADER CỦA CARD ===== */}
                    <div className="bg-slate-50 p-5 border-b border-slate-200">
                        <div className="flex flex-wrap items-center justify-between gap-4">
                            {/* Tiêu đề bên trái */}
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-indigo-100 text-indigo-600 rounded-lg flex items-center justify-center">
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                    </svg>
                                </div>
                                <h2 className="text-xl font-bold text-slate-800">
                                    Câu hỏi tương đồng
                                </h2>
                            </div>
                            {/* Các thông tin phụ bên phải */}
                            <div className="flex items-center gap-3 text-sm">
                                <div className="bg-slate-200 text-slate-700 px-3 py-1 rounded-full">
                                    Ngưỡng ≥ {threshold}
                                </div>
                                <div className="bg-indigo-100 text-indigo-700 px-3 py-1 rounded-full font-semibold">
                                    {details.similar_questions.length} kết quả
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ===== BODY CỦA CARD (CHỨA DANH SÁCH) ===== */}
                    <div className="p-2">
                        {details.similar_questions.length > 0 ? (
                            <div className="max-h-[70vh] overflow-y-auto custom-scrollbar">
                                {/* Chúng ta dùng <ul> để đúng ngữ nghĩa hơn */}
                                <ul className="divide-y divide-slate-200/80">
                                    {details.similar_questions.map((item) => (
                                        <li
                                            key={item.id}
                                            className="p-4 hover:bg-slate-50 transition-colors duration-200"
                                        >
                                            <div className="flex flex-col sm:flex-row sm:items-start gap-4">
                                                {/* Nội dung chính */}
                                                <div className="flex-1 space-y-3">
                                                    <p className="text-slate-800 leading-relaxed">
                                                        <Link
                                                            to={`/details/${item.id}`}
                                                            state={{ threshold }}
                                                            className="font-semibold text-indigo-600 hover:underline"
                                                        >
                                                            ID #{item.id}:
                                                        </Link>
                                                        <span className="ml-2 text-slate-700">{item.question}</span>
                                                    </p>
                                                    <div className="flex items-center gap-4 pt-1">
                                                        <span className="text-xs font-medium text-slate-500 w-24 flex-shrink-0">
                                                            Similarity Score
                                                        </span>
                                                        {/* Component SimilarityBar không thay đổi, vẫn hoạt động tốt */}
                                                        <SimilarityBar score={item.similarity} />
                                                    </div>
                                                </div>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ) : (
                            <div className="text-center py-16">
                                <div className="max-w-md mx-auto">
                                    <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                                        <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.5-.98-6.131-2.709M15 11V9a6 6 0 00-12 0v2" />
                                        </svg>
                                    </div>
                                    <h3 className="text-xl font-semibold text-gray-700 mb-2">Không tìm thấy kết quả</h3>
                                    <p className="text-gray-500 mb-4">Không có câu hỏi nào tương đồng với ngưỡng hiện tại.</p>
                                    <p className="text-sm text-gray-400">Hãy thử giảm ngưỡng ở trang báo cáo chính và xem lại.</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <style jsx>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 6px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: #f1f5f9;
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: linear-gradient(to bottom, #6366f1, #8b5cf6);
                    border-radius: 10px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(to bottom, #4f46e5, #7c3aed);
                }
                .animate-reverse-spin {
                    animation: reverse-spin 1s linear infinite;
                }
                @keyframes reverse-spin {
                    from {
                        transform: rotate(360deg);
                    }
                    to {
                        transform: rotate(0deg);
                    }
                }
            `}</style>
        </div>
    );
}

export default QuestionDetailPage;