// script.js
// DOM Elements
const originInput = document.getElementById('origin-input');
const destinationInput = document.getElementById('destination-input');
const requestBtn = document.getElementById('request-btn');
const cancelBtn = document.getElementById('cancel-btn');
const rideDetails = document.getElementById('ride-details');
const rideStatus = document.getElementById('ride-status');
const progressBar = document.getElementById('progress-bar');
const notification = document.getElementById('notification');
const originSuggestions = document.getElementById('origin-suggestions');
const destinationSuggestions = document.getElementById('destination-suggestions');
const locateBtn = document.getElementById('locate-btn');
const clearBtn = document.getElementById('clear-btn');

// Map variables
let map, originMarker, destinationMarker, routeLine;
let currentLocation = [35.972447, 50.732428]; // Default to London

// Initialize the map
function initMap() {
    map = L.map('map').setView(currentLocation, 15);
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Add location control
    locateBtn.addEventListener('click', () => {
        map.locate({setView: true, maxZoom: 16});
    });
    
    // Handle location found
    map.on('locationfound', function(e) {
        currentLocation = [e.latlng.lat, e.latlng.lng];
        if (!originMarker) {
            originMarker = L.marker(e.latlng).addTo(map)
                .bindPopup('Your location').openPopup();
            originInput.value = "Your current location";
            updateRideDetails();
        }
    });
    
    // Clear markers
    clearBtn.addEventListener('click', () => {
        if (originMarker) map.removeLayer(originMarker);
        if (destinationMarker) map.removeLayer(destinationMarker);
        if (routeLine) map.removeLayer(routeLine);
        originMarker = null;
        destinationMarker = null;
        originInput.value = "";
        destinationInput.value = "";
        rideDetails.style.display = 'none';
        originSuggestions.style.display = 'none';
        destinationSuggestions.style.display = 'none';
    });
    
    // Set up map click events
    map.on('click', function(e) {
        if (!originMarker) {
            originMarker = L.marker(e.latlng).addTo(map)
                .bindPopup('Pickup Location').openPopup();
            originInput.value = `Location (${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)})`;
        } else if (!destinationMarker) {
            destinationMarker = L.marker(e.latlng).addTo(map)
                .bindPopup('Destination').openPopup();
            destinationInput.value = `Location (${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)})`;
            
            // Draw route
            drawRoute(originMarker.getLatLng(), destinationMarker.getLatLng());
            updateRideDetails();
        }
    });
}

// Draw route on map
function drawRoute(start, end) {
    if (routeLine) map.removeLayer(routeLine);
    
    // In a real app, you would use a routing service like OSRM
    // Here we'll simulate a route with a straight line
    routeLine = L.polyline([start, end], {
        color: '#4361ee',
        weight: 4,
        opacity: 0.7,
        dashArray: '10, 10'
    }).addTo(map);
    
    // Fit map to route bounds
    map.fitBounds([start, end], {padding: [50, 50]});
}

// Update ride details
function updateRideDetails() {
    if (originMarker && destinationMarker) {
        const start = originMarker.getLatLng();
        const end = destinationMarker.getLatLng();
        
        // Calculate distance (simplified)
        const distance = Math.sqrt(
            Math.pow(end.lat - start.lat, 2) + 
            Math.pow(end.lng - start.lng, 2)
        ) * 100; // Rough km conversion
        
        const time = Math.round(distance * 3); // 3 min per km
        const price = (distance * 1.5 + 3).toFixed(2); // Base fare + per km
        
        document.getElementById('distance-value').textContent = distance.toFixed(1) + ' km';
        document.getElementById('time-value').textContent = time + ' min';
        document.getElementById('price-value').textContent = '$' + price;
        
        rideDetails.style.display = 'block';
    }
}

// Show notification
function showNotification(message, isSuccess = true) {
    notification.querySelector('span').textContent = message;
    notification.className = isSuccess ? 
        'notification success show' : 'notification error show';
    
    // Change icon based on success/error
    const icon = notification.querySelector('i');
    icon.className = isSuccess ? 'fas fa-check-circle' : 'fas fa-exclamation-circle';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    
    // Event listeners
    originInput.addEventListener('focus', function() {
        showLocationSuggestions('origin');
    });
    
    destinationInput.addEventListener('focus', function() {
        showLocationSuggestions('destination');
    });
    
    requestBtn.addEventListener('click', requestRide);
    cancelBtn.addEventListener('click', cancelRide);
});

// Show location suggestions
function showLocationSuggestions(type) {
    const suggestions = [
        {name: "Central Station", address: "Downtown, Main Street"},
        {name: "City Mall", address: "Shopping District"},
        {name: "Tech Park", address: "Innovation Road"},
        {name: "University Campus", address: "North Side"},
        {name: "Riverfront Park", address: "Riverside"},
        {name: "International Airport", address: "Airport Road"}
    ];
    
    const container = type === 'origin' ? originSuggestions : destinationSuggestions;
    container.innerHTML = '';
    
    suggestions.forEach(location => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.innerHTML = `
            <div class="suggestion-name">${location.name}</div>
            <div class="suggestion-address">${location.address}</div>
        `;
        
        item.addEventListener('click', () => {
            const input = type === 'origin' ? originInput : destinationInput;
            input.value = location.name;
            container.style.display = 'none';
            
            // Simulate setting a location
            if (type === 'origin') {
                if (originMarker) map.removeLayer(originMarker);
                const lat = 51.50 + (Math.random() * 0.1 - 0.05);
                const lng = -0.09 + (Math.random() * 0.1 - 0.05);
                originMarker = L.marker([lat, lng]).addTo(map)
                    .bindPopup('Pickup Location').openPopup();
            } else {
                if (destinationMarker) map.removeLayer(destinationMarker);
                const lat = 51.50 + (Math.random() * 0.1 - 0.05);
                const lng = -0.09 + (Math.random() * 0.1 - 0.05);
                destinationMarker = L.marker([lat, lng]).addTo(map)
                    .bindPopup('Destination').openPopup();
            }
            
            if (originMarker && destinationMarker) {
                drawRoute(originMarker.getLatLng(), destinationMarker.getLatLng());
                updateRideDetails();
            }
        });
        
        container.appendChild(item);
    });
    
    container.style.display = 'block';
}

// Request a ride
function requestRide() {
    if (!originMarker || !destinationMarker) {
        showNotification('Please select both origin and destination locations', false);
        return;
    }
    
    // Show loading state
    requestBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Finding a driver...';
    requestBtn.disabled = true;
    
    // Simulate API request to backend
    setTimeout(() => {
        // Hide request button and show cancel button
        requestBtn.style.display = 'none';
        cancelBtn.style.display = 'flex';
        
        // Show ride status
        rideStatus.style.display = 'block';
        
        // Update progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(interval);
                showNotification('Your driver has arrived!');
            }
        }, 1000);
        
        // Show success notification
        showNotification('Ride requested successfully! Driver assigned.');
    }, 2000);
}

// Cancel ride
function cancelRide() {
    // Reset UI
    requestBtn.style.display = 'flex';
    requestBtn.innerHTML = '<i class="fas fa-car"></i> Request Ride';
    requestBtn.disabled = false;
    cancelBtn.style.display = 'none';
    rideStatus.style.display = 'none';
    progressBar.style.width = '0%';
    
    showNotification('Ride cancelled successfully');
}

// Close suggestions when clicking outside
document.addEventListener('click', function(e) {
    if (!originInput.contains(e.target) && !originSuggestions.contains(e.target)) {
        originSuggestions.style.display = 'none';
    }
    if (!destinationInput.contains(e.target) && !destinationSuggestions.contains(e.target)) {
        destinationSuggestions.style.display = 'none';
    }
});