from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import RideBookingForm


@login_required
def create_ride(request):
    if request.method == 'POST':
        form = RideBookingForm(request.POST)
        if form.is_valid():
            # In a real app, you would save the ride request here
            # ride = form.save(commit=False)
            # ride.user = request.user
            # ride.save()
            return redirect('ride:confirm')
    else:
        form = RideBookingForm()

    return render(request, 'ride/create.html', {'form': form})


@login_required
def confirm_ride(request):
    return render(request, 'ride/confirm.html')
