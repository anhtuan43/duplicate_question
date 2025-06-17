# api_app/data/questions.py

import json
import os

# Đường dẫn tới file JSON chứa danh sách câu hỏi
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "questions.json")

def load_questions(filepath):
    """Đọc danh sách câu hỏi từ file JSON"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ALL_QUESTIONS được load từ file
ALL_QUESTIONS = load_questions(QUESTIONS_FILE)
