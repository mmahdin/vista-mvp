# schedule/urls.py
from django.urls import path
from . import views

app_name = 'schedule'

urlpatterns = [
    path('create/', views.create_scheduled_ride, name='create'),
    # Add other scheduling paths here
]
