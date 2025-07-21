from .models import RideRequest
from django.contrib.auth.models import User
from django.utils.timezone import now
from datetime import timedelta


def match_rides_to_cars():
    """
    Simulate car assignment by grouping users with similar routes.
    """
    pending_requests = RideRequest.objects.filter(status='pending')

    if not pending_requests.exists():
        return "No pending ride requests."

    for request in pending_requests:
        # For now, simulate driver assignment (can expand later)
        fake_driver = User.objects.filter(is_staff=True).first()
        if fake_driver:
            request.assigned_driver = fake_driver
            request.status = 'assigned'
            request.save()

    return "Matching complete."
