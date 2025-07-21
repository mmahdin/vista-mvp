from django.urls import path
from . import views

app_name = 'ride'


urlpatterns = [
    path('book-ride/', views.ride_booking_view, name='book_ride'),
]
