from django.urls import path
from . import views  # Import the views from your app

urlpatterns = [
    path('', views.index, name='index'),  # The home page (index) where everything is rendered
    path('video_feed/', views.video_feed, name='video_feed'),  # URL for the camera video stream
    path('stop_camera/', views.stop_camera, name='stop_camera'),  # URL to stop the camera
    path('detected_student_info/', views.detected_student_info, name='detected_student_info'),  # URL for AJAX call to get the detected student info
    path('train/', views.train, name='train'),  # URL for submitting student training information
    path('submit-data', views.train, name='submit_data'),
    path('students/<int:student_id>/', views.student_details, name='student_details'),
]
