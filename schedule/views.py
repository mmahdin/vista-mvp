# schedule/views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def create_scheduled_ride(request):
    return render(request, 'schedule/create.html')
