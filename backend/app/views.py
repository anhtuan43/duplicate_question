# api_app/views.py

import json
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import torch

# Import AppConfig để truy cập detector đã được khởi tạo
from .apps import ApiAppConfig
from .detector import DuplicateAnalysisResult # Import để type hinting

def format_analysis_result(result: DuplicateAnalysisResult, questions: list) -> dict:
    """Hàm trợ giúp để định dạng lại kết quả từ detector sang JSON response."""
    stats = result.statistics
    
    total_possible_pairs = stats['total_questions'] * (stats['total_questions'] - 1) // 2
    if total_possible_pairs == 0: total_possible_pairs = 1

    sim_dist_list = [
        {
            "range": range_name,
            "pairs": count,
            "percentage": round((count / total_possible_pairs) * 100, 2)
        }
        for range_name, count in stats['similarity_distribution'].items()
    ]
    
    sim_stats_obj = {
        "max": round(stats.get('max_similarity', 0), 4),
        "min": round(stats.get('min_similarity', 0), 4),
        "avg": round(stats.get('avg_similarity', 0), 4)
    }
    
    top_dup_list = [
        {
            "id": idx,
            "duplicates": count,
            "question": questions[idx]
        }
        for idx, count in stats.get('most_duplicated_questions', [])
    ]
    
    return {
        "totalQuestions": stats['total_questions'],
        "uniqueQuestions": stats['unique_questions'],
        "duplicateQuestions": stats['duplicate_questions'],
        "duplicateRate": round(stats['duplicate_rate'], 2),
        "duplicatePairs": stats['duplicate_pairs'],
        "similarityDistribution": sim_dist_list,
        "similarityStats": sim_stats_obj,
        "topDuplicates": top_dup_list
    }


@csrf_exempt
@require_http_methods(["POST"])
def analyze_preloaded_view(request: HttpRequest):
    """
    View phân tích danh sách câu hỏi đã được tải sẵn trên server.
    Chỉ nhận 'threshold' từ request body.
    """
    try:
        data = json.loads(request.body)
        threshold = data.get('threshold')

        if threshold is None:
            return JsonResponse({'error': 'Thiếu tham số "threshold".'}, status=400)
            
        if not (0.0 <= float(threshold) <= 1.0):
             return JsonResponse({'error': 'Threshold phải là một số từ 0.0 đến 1.0.'}, status=400)
            
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Dữ liệu đầu vào không hợp lệ hoặc sai định dạng.'}, status=400)

    try:
        detector = ApiAppConfig.detector
        # Lấy danh sách câu hỏi đã được tải sẵn
        questions = ApiAppConfig.preloaded_questions

        if not questions:
            return JsonResponse({'error': 'Không tìm thấy dữ liệu câu hỏi trên server.'}, status=500)

        # Chạy phân tích với dữ liệu và ngưỡng đã cho
        analysis_result = detector.analyze_duplicates(
            questions=questions,
            threshold=threshold,
            return_similarity_matrix=False 
        )
        
        # Định dạng và trả về kết quả
        formatted_response = format_analysis_result(analysis_result, questions)
        return JsonResponse(formatted_response, status=200)

    except Exception as e:
        print(f"Lỗi hệ thống khi phân tích: {e}")
        return JsonResponse({'error': f'Đã xảy ra lỗi phía server: {str(e)}'}, status=500)
    

@csrf_exempt
@require_http_methods(["POST"])
def question_details_view(request: HttpRequest, question_id: int):
    """
    View trả về chi tiết tương đồng cho một câu hỏi cụ thể.
    Nhận 'threshold' từ request body.
    """
    try:
        data = json.loads(request.body)
        threshold = data.get('threshold')

        if threshold is None or not (0.0 <= float(threshold) <= 1.0):
            return JsonResponse({'error': 'Threshold hợp lệ (0.0-1.0) là bắt buộc.'}, status=400)
            
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({'error': 'Dữ liệu đầu vào không hợp lệ.'}, status=400)

    try:
        detector = ApiAppConfig.detector
        questions = ApiAppConfig.preloaded_questions

        if not (0 <= question_id < len(questions)):
            return JsonResponse({'error': 'Question ID không hợp lệ.'}, status=404)

        # 1. Lấy embeddings (sẽ rất nhanh nếu đã có trong cache)
        embeddings = detector.encode_questions(questions)
        
        # 2. Lấy vector của câu hỏi chính
        target_embedding = embeddings[question_id]
        
        # 3. Tính độ tương đồng của câu hỏi chính với TẤT CẢ các câu khác
        # Sử dụng phép nhân vector-ma trận (torch.mv) để tối ưu hiệu suất
        with torch.no_grad():
            all_similarities = torch.mv(embeddings, target_embedding)

        # 4. Lọc và định dạng kết quả
        similar_pairs = []
        for i, score in enumerate(all_similarities.cpu().numpy()):
            # Bỏ qua việc so sánh với chính nó
            if i == question_id:
                continue
            
            if score >= threshold:
                similar_pairs.append({
                    "id": i,
                    "question": questions[i],
                    "similarity": round(float(score), 4)
                })
        
        # Sắp xếp theo độ tương đồng giảm dần
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 5. Đóng gói response
        response_data = {
            "main_question": {
                "id": question_id,
                "text": questions[question_id]
            },
            "similar_questions": similar_pairs
        }
        
        return JsonResponse(response_data, status=200)

    except Exception as e:
        print(f"Lỗi khi lấy chi tiết câu hỏi: {e}")
        return JsonResponse({'error': f'Đã xảy ra lỗi phía server: {str(e)}'}, status=500)