# history/urls.py
from django.urls import path
from . import views

app_name = 'history'

urlpatterns = [
    path('', views.ride_history, name='list'),
    # Add other history paths here
]
