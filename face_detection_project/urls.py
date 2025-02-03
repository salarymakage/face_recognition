from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detection.urls')),  # Ensure detection app's URLs are included
]
