from django.urls import path
from . import views

app_name = 'ride'

urlpatterns = [
    path('create/', views.create_ride, name='create'),
    path('confirm/', views.confirm_ride, name='confirm'),
]
