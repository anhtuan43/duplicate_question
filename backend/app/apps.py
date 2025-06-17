from django.apps import AppConfig

# Import các class cần thiết
from .detector import QuestionDuplicateDetector, DuplicateDetectorConfig
from .data.questions import ALL_QUESTIONS

class ApiAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    # Khai báo một biến class để giữ detector
    detector = None

    # Khai báo các biến class
    detector = None
    preloaded_questions = None # Biến mới để giữ câu hỏi

    def ready(self):
        """Hàm này được gọi một lần duy nhất khi ứng dụng sẵn sàng."""
        
        # Tải model (nếu chưa có)
        if not self.detector:
            print("🚀 Đang khởi tạo và tải model Duplicate Detector...")
            config = DuplicateDetectorConfig(batch_size=64, cache_embeddings=True)
            ApiAppConfig.detector = QuestionDuplicateDetector(config)
            print("✅ Model đã sẵn sàng!")
        
        # --- THAY ĐỔI Ở ĐÂY ---
        # Tải danh sách câu hỏi vào bộ nhớ (nếu chưa có)
        if not self.preloaded_questions:
            print(f"📚 Đang tải {len(ALL_QUESTIONS)} câu hỏi vào bộ nhớ...")
            ApiAppConfig.preloaded_questions = ALL_QUESTIONS
            print("✅ Dữ liệu câu hỏi đã sẵn sàng!")
        # --- KẾT THÚC THAY ĐỔI ---