import React, { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';

// Component Icon Sắp xếp
const SortIcon = ({ direction }) => {
    if (!direction) {
        // Trạng thái chờ, chưa sắp xếp
        return <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" /></svg>;
    }
    // Đã sắp xếp
    return direction === 'asc' 
        ? <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z" clipRule="evenodd" /></svg>
        : <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" /></svg>;
};

// Component Phân trang
const Pagination = ({ currentPage, totalPages, onPageChange }) => {
    if (totalPages <= 1) return null;

    const pageNumbers = [];
    for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i);
    }

    return (
        <div className="flex items-center justify-center gap-2">
            <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="px-3 py-1 rounded-md bg-white border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                Trước
            </button>
            {pageNumbers.map(number => (
                <button
                    key={number}
                    onClick={() => onPageChange(number)}
                    className={`w-8 h-8 rounded-md ${currentPage === number ? 'bg-blue-600 text-white font-bold' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'}`}
                >
                    {number}
                </button>
            ))}
            <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="px-3 py-1 rounded-md bg-white border border-slate-300 text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
                Sau
            </button>
        </div>
    );
};


function TopDuplicatesTable({ data, currentThreshold }) {
    // --- STATE ---
    const [sortConfig, setSortConfig] = useState({ key: 'duplicates', direction: 'desc' });
    const [currentPage, setCurrentPage] = useState(1);
    const ITEMS_PER_PAGE = 10;

    // --- LOGIC SẮP XẾP VÀ PHÂN TRANG ---
    const sortedData = useMemo(() => {
        const sortableData = [...data];
        if (sortConfig.key) {
            sortableData.sort((a, b) => {
                if (a[sortConfig.key] < b[sortConfig.key]) {
                    return sortConfig.direction === 'asc' ? -1 : 1;
                }
                if (a[sortConfig.key] > b[sortConfig.key]) {
                    return sortConfig.direction === 'asc' ? 1 : -1;
                }
                return 0;
            });
        }
        return sortableData;
    }, [data, sortConfig]);

    const currentTableData = useMemo(() => {
        const firstPageIndex = (currentPage - 1) * ITEMS_PER_PAGE;
        const lastPageIndex = firstPageIndex + ITEMS_PER_PAGE;
        return sortedData.slice(firstPageIndex, lastPageIndex);
    }, [currentPage, sortedData]);

    const totalPages = Math.ceil(data.length / ITEMS_PER_PAGE);

    // --- HÀM XỬ LÝ ---
    const handleSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
        setCurrentPage(1); // Quay về trang đầu khi sắp xếp
    };

    const getBadgeStyle = (count) => {
        if (count >= 10) return 'bg-red-100 text-red-800';
        if (count >= 5) return 'bg-orange-100 text-orange-800';
        if (count >= 3) return 'bg-yellow-100 text-yellow-800';
        return 'bg-green-100 text-green-800';
    };

    const tableHeaders = [
        { key: 'id', label: 'ID' },
        { key: 'duplicates', label: 'Duplicates' },
        { key: 'question', label: 'Câu hỏi' },
    ];

    // --- RENDER ---
    return (
        <div className="bg-white rounded-xl shadow-lg border border-slate-200/80">
            {/* Header */}
            <div className="p-5 border-b border-slate-200">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <h3 className="text-lg font-bold text-slate-800">
                        Top Câu Hỏi Duplicate
                    </h3>
                    <div className="flex items-center gap-4 text-sm text-slate-600">
                        <span>Threshold: <span className="font-semibold text-slate-900">{currentThreshold}</span></span>
                        <span>Tổng: <span className="font-semibold text-slate-900">{data.length} câu hỏi</span></span>
                    </div>
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-slate-600">
                        <tr>
                            {tableHeaders.map(({ key, label }) => (
                                <th key={key} scope="col" className="px-6 py-3 text-left font-medium">
                                    <div 
                                        className="flex items-center gap-2 cursor-pointer"
                                        onClick={() => handleSort(key)}
                                    >
                                        {label.toUpperCase()}
                                        <SortIcon direction={sortConfig.key === key ? sortConfig.direction : null} />
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {currentTableData.map((item) => (
                            <tr key={item.id} className="hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-700">
                                    #{item.id}
                                </td>
                                <td className="px-6 py-4">
                                    <span className={`px-2.5 py-0.5 rounded-full font-semibold ${getBadgeStyle(item.duplicates)}`}>
                                        {item.duplicates}
                                    </span>
                                </td>
                                <td className="px-6 py-4 max-w-md">
                                    <Link
                                        to={`/details/${item.id}`}
                                        state={{ threshold: currentThreshold }}
                                        className="font-semibold text-indigo-600 hover:underline"
                                    >
                                        {item.question}
                                    </Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer với Phân trang */}
            <div className="p-5 border-t border-slate-200 flex flex-wrap items-center justify-between gap-4">
                <div className="text-sm text-slate-600">
                    Hiển thị <span className="font-semibold">{currentTableData.length}</span> trong tổng số <span className="font-semibold">{data.length}</span> kết quả
                </div>
                <Pagination
                    currentPage={currentPage}
                    totalPages={totalPages}
                    onPageChange={setCurrentPage}
                />
            </div>
        </div>
    );
}

export default TopDuplicatesTable;