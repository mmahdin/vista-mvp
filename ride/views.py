from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def create_ride(request):
    return render(request, 'ride/create.html')
