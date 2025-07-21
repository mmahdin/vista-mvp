# history/views.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def ride_history(request):
    return render(request, 'history/list.html')
