from django.urls import path
from .views import start_recording_view, check_status_view, emotion_recognition

urlpatterns = [
    path('start_recording/', start_recording_view, name='start_recording'),
    path('check_status/', check_status_view, name='check_status'),
    path('', emotion_recognition, name='emotion_recognition'),
]
