from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required  # Ensures only logged-in users can access this page
def ride_booking_view(request):
    """
    Renders the ride booking page with the interactive map.
    """
    # You can pass any context data needed by your template here,
    # for example, user-specific settings or initial map coordinates.
    context = {
        'page_title': 'Book Your Ride',
    }
    return render(request, 'ride/ride_app.html', context)
