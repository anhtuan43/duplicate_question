// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DuplicateAnalysisInterface from './DuplicateAnalysisInterface';
import QuestionDetailPage from './QuestionDetailPage'; // Component mới sẽ tạo ở bước sau

function App() {
    return (
        <Router>
            <Routes>
                {/* Route cho trang báo cáo chính */}
                <Route path="/" element={<DuplicateAnalysisInterface />} />
                
                {/* Route cho trang chi tiết, với tham số động là questionId */}
                <Route path="/details/:questionId" element={<QuestionDetailPage />} />
            </Routes>
        </Router>
    );
}

export default App;