# api_app/urls.py

from django.urls import path
from .views import analyze_preloaded_view, question_details_view

urlpatterns = [
    path('analyze/', analyze_preloaded_view, name='analyze_questions'),
    path('question-details/<int:question_id>/', question_details_view, name='question_details'),    
]