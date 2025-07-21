from django.contrib import admin
from .models import RideRequest


@admin.register(RideRequest)
class RideRequestAdmin(admin.ModelAdmin):
    list_display = ('user', 'origin_lat', 'origin_lng',
                    'destination_lat', 'destination_lng', 'status', 'requested_at')
    list_filter = ('status', 'requested_at')
