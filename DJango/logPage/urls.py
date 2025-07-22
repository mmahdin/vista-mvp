from django.urls import path
from . import views


app_name = 'logPage'


urlpatterns = [
    path('', views.index, name="index"),
    path('register.html', views.register, name="register"),
    path('forgot-password.html', views.forget, name="forget"),
    path('registerUser/', views.addUser, name="registerUser"),
    path('verifyUser/', views.verifyUser, name="authenticate"),
]
