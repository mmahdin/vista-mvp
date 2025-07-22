from django.urls import path
from . import views

app_name = 'ride'


urlpatterns = [
    path('book-ride/', views.ride_booking_view, name='book_ride'),
    path('api/submit/', views.submit_ride_request, name='submit_ride_request'),
    path('api/status/', views.check_ride_status, name='check_ride_status'),
]
