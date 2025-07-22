from django.http import JsonResponse
import json
from .models import RideRequest
from django.views.decorators.csrf import csrf_exempt
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


@csrf_exempt
@login_required
def submit_ride_request(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        ride = RideRequest.objects.create(
            user=request.user,
            origin_lat=data['origin_lat'],
            origin_lng=data['origin_lng'],
            destination_lat=data['destination_lat'],
            destination_lng=data['destination_lng'],
        )
        return JsonResponse({'status': 'success', 'ride_id': ride.id})
    return JsonResponse({'error': 'Invalid method'}, status=405)


@login_required
def check_ride_status(request):
    ride = RideRequest.objects.filter(user=request.user, status__in=[
                                      'pending', 'assigned']).last()
    if ride:
        return JsonResponse({
            'status': ride.status,
            'driver': ride.assigned_driver.username if ride.assigned_driver else None
        })
    return JsonResponse({'status': 'none'})
