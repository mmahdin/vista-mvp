// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================
const CONFIG = {
    DEFAULT_LOCATION: [35.972447, 50.732428],
    SUGGESTIONS_DEBOUNCE_TIME: 300,
    DISTANCE_MULTIPLIER: 100,
    TIME_PER_KM: 3,
    BASE_FARE: 3,
    PRICE_PER_KM: 1.5,
    NOTIFICATION_DURATION: 3000,
    RIDE_REQUEST_DELAY: 2000,
    PROGRESS_INTERVAL: 1000,
    PROGRESS_INCREMENT: 5
};

const ICONS = {
    green: L.icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        shadowSize: [41, 41]
    }),
    red: L.icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41],
        iconAnchor: [12, 41],
        shadowSize: [41, 41]
    })
};

const LOCATION_ICONS = {
    restaurant: 'fa-utensils',
    cafe: 'fa-utensils',
    hotel: 'fa-hotel',
    tourism: 'fa-hotel',
    station: 'fa-train',
    railway: 'fa-train',
    airport: 'fa-plane',
    shop: 'fa-shopping-cart',
    default: 'fa-map-marker-alt'
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
class Utils {
    static escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    static calculateDistance(start, end) {
        return Math.sqrt(
            Math.pow(end.lat - start.lat, 2) + 
            Math.pow(end.lng - start.lng, 2)
        ) * CONFIG.DISTANCE_MULTIPLIER;
    }

    static calculateTime(distance) {
        return Math.round(distance * CONFIG.TIME_PER_KM);
    }

    static calculatePrice(distance) {
        return (distance * CONFIG.PRICE_PER_KM + CONFIG.BASE_FARE).toFixed(2);
    }

    static getLocationIcon(item) {
        if (item.type && LOCATION_ICONS[item.type]) {
            return LOCATION_ICONS[item.type];
        }
        if (item.class && LOCATION_ICONS[item.class]) {
            return LOCATION_ICONS[item.class];
        }
        return LOCATION_ICONS.default;
    }

    static parseLocationName(displayName) {
        const nameParts = displayName ? displayName.split(',') : ['Unknown location'];
        const locationName = nameParts[0] || 'Unknown location';
        const locationAddress = nameParts.slice(1, 3).join(',').trim() || 'No address available';
        return { locationName, locationAddress };
    }
}

// ============================================================================
// GEOCODING SERVICE
// ============================================================================
class GeocodingService {
    constructor() {
        this.currentRequest = null;
    }

    async reverseGeocode(coords) {
        const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${coords[0]}&lon=${coords[1]}`;
        
        try {
            const response = await fetch(url);
            const data = await response.json();
            return data.display_name || null;
        } catch (error) {
            console.error('Reverse geocode error:', error);
            return null;
        }
    }

    async fetchLocationSuggestions(query) {
        if (!query || query.length < 3) {
            return [];
        }

        // Cancel any pending request
        if (this.currentRequest) {
            this.currentRequest.abort();
        }

        const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`;
        
        // Create new AbortController for this request
        const controller = new AbortController();
        this.currentRequest = controller;

        try {
            const response = await fetch(url, { signal: controller.signal });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data || [];
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Request aborted');
                return [];
            }
            console.error('Fetch error:', error);
            throw error;
        } finally {
            this.currentRequest = null;
        }
    }
}

// ============================================================================
// NOTIFICATION SYSTEM
// ============================================================================
class NotificationManager {
    constructor(notificationElement) {
        this.notification = notificationElement;
    }

    show(message, isSuccess = true) {
        // Reset notification classes
        this.notification.className = 'notification';
        
        // Add appropriate classes
        this.notification.classList.add(isSuccess ? 'success' : 'error', 'show');
        
        // Update content
        this.notification.querySelector('.notification-message').textContent = message;
        
        // Hide after configured duration
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, CONFIG.NOTIFICATION_DURATION);
    }
}

// ============================================================================
// SUGGESTIONS MANAGER
// ============================================================================
class SuggestionsManager {
    constructor(geocodingService, mapManager) {
        this.geocodingService = geocodingService;
        this.mapManager = mapManager;
        this.debounceTimer = null;
    }

    async displaySuggestions(query, type, container) {
        if (!query || query.length < 3) {
            container.innerHTML = '';
            container.style.display = 'none';
            return;
        }

        container.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Searching locations...</div>';
        container.style.display = 'block';

        try {
            const suggestions = await this.geocodingService.fetchLocationSuggestions(query);
            this.renderSuggestions(suggestions, type, container);
        } catch (error) {
            container.innerHTML = '<div class="suggestion-item error">Error loading suggestions. Please try again.</div>';
        }
    }

    renderSuggestions(suggestions, type, container) {
        container.innerHTML = '';

        if (!suggestions || suggestions.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'suggestion-item no-results';
            noResults.textContent = 'No results found';
            container.appendChild(noResults);
            return;
        }

        const fragment = document.createDocumentFragment();

        suggestions.forEach((item, index) => {
            const suggestionItem = this.createSuggestionItem(item, index, type);
            fragment.appendChild(suggestionItem);
        });

        container.appendChild(fragment);
    }

    createSuggestionItem(item, index, type) {
        const suggestionItem = document.createElement('div');
        suggestionItem.className = 'suggestion-item';
        suggestionItem.setAttribute('data-index', index);
        suggestionItem.setAttribute('tabindex', '0');

        const iconClass = Utils.getLocationIcon(item);
        const { locationName, locationAddress } = Utils.parseLocationName(item.display_name);

        suggestionItem.innerHTML = `
            <div class="suggestion-icon">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="suggestion-text">
                <div class="suggestion-name">${Utils.escapeHtml(locationName)}</div>
                <div class="suggestion-address">${Utils.escapeHtml(locationAddress)}</div>
            </div>
        `;

        this.attachSuggestionEventListeners(suggestionItem, item, type);
        return suggestionItem;
    }

    attachSuggestionEventListeners(suggestionItem, item, type) {
        const clickHandler = (event) => {
            event.preventDefault();
            event.stopPropagation();
            this.handleSuggestionClick(item, type);
        };

        suggestionItem.addEventListener('click', clickHandler);
        suggestionItem.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                clickHandler(event);
            }
        });

        // Add hover effects
        suggestionItem.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f0f0f0';
        });

        suggestionItem.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    }

    handleSuggestionClick(item, type) {
        try {
            const input = type === 'origin' ? 
                document.getElementById('origin-input') : 
                document.getElementById('destination-input');
            
            const container = type === 'origin' ? 
                document.getElementById('origin-suggestions') : 
                document.getElementById('destination-suggestions');

            if (!input) {
                console.error('Input element not found for type:', type);
                return;
            }

            input.value = item.display_name || '';
            container.style.display = 'none';

            const lat = parseFloat(item.lat);
            const lng = parseFloat(item.lon);

            if (isNaN(lat) || isNaN(lng)) {
                console.error('Invalid coordinates:', item.lat, item.lon);
                return;
            }

            this.mapManager.addMarker([lat, lng], type);
        } catch (error) {
            console.error('Error handling suggestion click:', error);
        }
    }

    debouncedFetch(query, type, container) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => {
            this.displaySuggestions(query, type, container);
        }, CONFIG.SUGGESTIONS_DEBOUNCE_TIME);
    }
}

// ============================================================================
// MAP MANAGER
// ============================================================================
class MapManager {
    constructor(mapElementId, rideDetailsManager, geocodingService) {
        this.mapElementId = mapElementId;
        this.rideDetailsManager = rideDetailsManager;
        this.geocodingService = geocodingService;
        this.map = null;
        this.originMarker = null;
        this.destinationMarker = null;
        this.routeLine = null;
        this.currentLocation = CONFIG.DEFAULT_LOCATION;
    }

    initialize() {
        this.map = L.map(this.mapElementId, {
            zoomControl: false,
            attributionControl: false
        }).setView(this.currentLocation, 16);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(this.map);

        L.control.zoom({
            position: 'bottomright'
        }).addTo(this.map);

        this.setupMapEventListeners();
    }

    setupMapEventListeners() {
        this.map.on('click', async (e) => {
            if (!this.originMarker) {
                this.addMarker([e.latlng.lat, e.latlng.lng], 'origin');
                
                const address = await this.geocodingService.reverseGeocode([e.latlng.lat, e.latlng.lng]);
                document.getElementById('origin-input').value = address || "Selected location";
            } else if (!this.destinationMarker) {
                this.addMarker([e.latlng.lat, e.latlng.lng], 'destination');
                
                const address = await this.geocodingService.reverseGeocode([e.latlng.lat, e.latlng.lng]);
                document.getElementById('destination-input').value = address || "Selected location";
                
                this.drawRoute();
                this.rideDetailsManager.update();
            }
        });
    }

    // Snap to road using OSRM API
    async snapToRoadOSRM(coords) {
        const osrmNearestUrl = `http://router.project-osrm.org/nearest/v1/driving/${coords[1]},${coords[0]}.json`;
        
        try {
            const response = await fetch(osrmNearestUrl);
            if (!response.ok) {
                throw new Error('OSRM nearest request failed');
            }
            
            const data = await response.json();
            if (!data.waypoints || data.waypoints.length === 0) {
                throw new Error('No nearest road found');
            }
            
            const nearestLon = data.waypoints[0].location[0];
            const nearestLat = data.waypoints[0].location[1];
            
            return [nearestLat, nearestLon];
        } catch (error) {
            console.error('OSRM snap to road error:', error);
            throw error;
        }
    }

    // Snap to road using Nominatim API
    async snapToRoadNominatim(coords) {
        const nominatimUrl = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${coords[0]}&lon=${coords[1]}&zoom=18&addressdetails=1`;
        
        try {
            const response = await fetch(nominatimUrl);
            if (!response.ok) {
                throw new Error('Nominatim reverse geocoding request failed');
            }
            
            const data = await response.json();
            if (!data || !data.address) {
                throw new Error('No address found for the given coordinates');
            }
            
            const nearestLat = parseFloat(data.lat);
            const nearestLon = parseFloat(data.lon);
            
            return [nearestLat, nearestLon];
        } catch (error) {
            console.error('Nominatim snap to road error:', error);
            throw error;
        }
    }

    // Main addMarker function with optional snap-to-road
    async addMarker(coords, type, snapToRoad = true, snapAPI = 'nominatim') {
        let finalCoords = coords;
        
        // Apply snap to road if enabled
        if (snapToRoad) {
            try {
                let snappedCoords;
                
                switch (snapAPI.toLowerCase()) {
                    case 'osrm':
                        snappedCoords = await this.snapToRoadOSRM(coords);
                        break;
                    case 'nominatim':
                        snappedCoords = await this.snapToRoadNominatim(coords);
                        break;
                    default:
                        console.warn(`Unknown snap API: ${snapAPI}. Using original coordinates.`);
                        snappedCoords = coords;
                }
                
                finalCoords = snappedCoords;
            } catch (error) {
                console.error('Error snapping to road:', error);
                console.log('Falling back to original coordinates');
                // Keep original coordinates as fallback
            }
        }
        
        const latLng = L.latLng(finalCoords[0], finalCoords[1]);
        
        // Handle marker placement based on type
        if (type === 'origin') {
            if (this.originMarker) {
                this.map.removeLayer(this.originMarker);
            }
            this.originMarker = L.marker(latLng, { icon: ICONS.green }).addTo(this.map);
            this.map.setView(latLng, 15);
        } else {
            if (this.destinationMarker) {
                this.map.removeLayer(this.destinationMarker);
            }
            this.destinationMarker = L.marker(latLng, { icon: ICONS.red }).addTo(this.map);
        }
        
        // Draw route if both markers exist
        if (this.originMarker && this.destinationMarker) {
            this.drawRoute();
            this.rideDetailsManager.update();
        }
    }

    drawRoute() {
        if (!this.originMarker || !this.destinationMarker) return;

        if (this.routeLine) {
            this.map.removeLayer(this.routeLine);
        }

        const start = this.originMarker.getLatLng();
        const end = this.destinationMarker.getLatLng();

        // Simulated route points (in a real app, use a routing service)
        const routePoints = [
            start,
            [start.lat + (end.lat - start.lat) * 0.3, start.lng + (end.lng - start.lng) * 0.3],
            [start.lat + (end.lat - start.lat) * 0.7, start.lng + (end.lng - start.lng) * 0.7],
            end
        ];

        this.routeLine = L.polyline(routePoints, {
            color: '#4361ee',
            weight: 5,
            opacity: 0.8,
            lineJoin: 'round',
            dashArray: '10, 10'
        }).addTo(this.map);

        // Fit map to route bounds
        this.map.fitBounds([start, end], { padding: [100, 100] });
    }

    clearMarkers() {
        if (this.originMarker) {
            this.map.removeLayer(this.originMarker);
            this.originMarker = null;
        }
        if (this.destinationMarker) {
            this.map.removeLayer(this.destinationMarker);
            this.destinationMarker = null;
        }
        if (this.routeLine) {
            this.map.removeLayer(this.routeLine);
            this.routeLine = null;
        }
    }

    getCurrentLocation(callback) {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const userLocation = [
                        position.coords.latitude,
                        position.coords.longitude
                    ];
                    
                    this.map.setView(userLocation, 15);
                    
                    if (this.originMarker) {
                        this.map.removeLayer(this.originMarker);
                    }
                    
                    this.originMarker = L.marker(userLocation, { icon: ICONS.green })
                        .addTo(this.map)
                        .bindPopup('Your location')
                        .openPopup();
                    
                    document.getElementById('origin-input').value = "Your current location";
                    
                    if (this.destinationMarker) {
                        this.drawRoute();
                        this.rideDetailsManager.update();
                    }
                    
                    callback(null, 'Location found!');
                },
                (error) => {
                    callback(error, null);
                }
            );
        } else {
            callback(new Error('Geolocation is not supported by your browser'), null);
        }
    }

    getMarkers() {
        return {
            origin: this.originMarker,
            destination: this.destinationMarker
        };
    }
}

// ============================================================================
// RIDE DETAILS MANAGER
// ============================================================================
class RideDetailsManager {
    constructor(mapManager) {
        this.mapManager = mapManager;
        this.rideDetailsElement = document.getElementById('ride-details');
    }

    update() {
        const markers = this.mapManager.getMarkers();
        
        if (markers.origin && markers.destination) {
            const start = markers.origin.getLatLng();
            const end = markers.destination.getLatLng();
            
            const distance = Utils.calculateDistance(start, end);
            const time = Utils.calculateTime(distance);
            const price = Utils.calculatePrice(distance);
            
            document.getElementById('distance-value').textContent = distance.toFixed(1) + ' km';
            document.getElementById('time-value').textContent = time + ' min';
            document.getElementById('price-value').textContent = '$' + price;
            
            this.rideDetailsElement.style.display = 'block';
        }
    }

    hide() {
        this.rideDetailsElement.style.display = 'none';
    }
}

// ============================================================================
// RIDE MANAGER
// ============================================================================
class RideManager {
    constructor(mapManager, notificationManager) {
        this.mapManager = mapManager;
        this.notificationManager = notificationManager;
        this.requestBtn = document.getElementById('request-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        this.rideStatus = document.getElementById('ride-status');
        this.progressBar = document.getElementById('progress-bar');
        this.progressInterval = null;
    }

    async requestRide() {
        const markers = this.mapManager.getMarkers();
        
        if (!markers.origin || !markers.destination) {
            this.notificationManager.show('Please select both origin and destination locations', false);
            return;
        }

        // Show loading state
        this.requestBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Finding a driver...';
        this.requestBtn.disabled = true;

        // Simulate API request
        setTimeout(() => {
            this.showRideInProgress();
            this.startProgressAnimation();
            this.notificationManager.show('Ride requested successfully! Driver assigned.');
        }, CONFIG.RIDE_REQUEST_DELAY);
    }

    showRideInProgress() {
        this.requestBtn.style.display = 'none';
        this.cancelBtn.style.display = 'flex';
        this.rideStatus.style.display = 'block';
    }

    startProgressAnimation() {
        let progress = 0;
        this.progressInterval = setInterval(() => {
            progress += CONFIG.PROGRESS_INCREMENT;
            this.progressBar.style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(this.progressInterval);
                this.notificationManager.show('Your driver has arrived!');
            }
        }, CONFIG.PROGRESS_INTERVAL);
    }

    cancelRide() {
        // Clear progress interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }

        // Reset UI
        this.requestBtn.style.display = 'flex';
        this.requestBtn.innerHTML = '<i class="fas fa-car"></i> Request Ride';
        this.requestBtn.disabled = false;
        this.cancelBtn.style.display = 'none';
        this.rideStatus.style.display = 'none';
        this.progressBar.style.width = '0%';
        
        this.notificationManager.show('Ride cancelled successfully');
    }
}

// ============================================================================
// MAIN APPLICATION CLASS
// ============================================================================
class RideApp {
    constructor() {
        this.geocodingService = new GeocodingService();
        this.notificationManager = new NotificationManager(document.getElementById('notification'));
        this.rideDetailsManager = null; // Will be initialized after mapManager
        this.mapManager = new MapManager('map', null, this.geocodingService);
        this.suggestionsManager = new SuggestionsManager(this.geocodingService, this.mapManager);
        this.rideManager = null; // Will be initialized after mapManager
        
        // Initialize ride details manager with map manager
        this.rideDetailsManager = new RideDetailsManager(this.mapManager);
        this.mapManager.rideDetailsManager = this.rideDetailsManager;
        
        // Initialize ride manager
        this.rideManager = new RideManager(this.mapManager, this.notificationManager);
        
        this.initializeEventListeners();
    }

    initialize() {
        this.mapManager.initialize();
    }

    initializeEventListeners() {
        // Input event listeners
        const originInput = document.getElementById('origin-input');
        const destinationInput = document.getElementById('destination-input');
        const originSuggestions = document.getElementById('origin-suggestions');
        const destinationSuggestions = document.getElementById('destination-suggestions');

        originInput.addEventListener('input', () => {
            this.suggestionsManager.debouncedFetch(originInput.value, 'origin', originSuggestions);
        });

        destinationInput.addEventListener('input', () => {
            this.suggestionsManager.debouncedFetch(destinationInput.value, 'destination', destinationSuggestions);
        });

        // Enter key listeners
        originInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && originInput.value.length >= 3) {
                this.suggestionsManager.displaySuggestions(originInput.value, 'origin', originSuggestions);
            }
        });

        destinationInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && destinationInput.value.length >= 3) {
                this.suggestionsManager.displaySuggestions(destinationInput.value, 'destination', destinationSuggestions);
            }
        });

        // Button event listeners
        document.getElementById('locate-btn').addEventListener('click', () => {
            this.mapManager.getCurrentLocation((error, message) => {
                if (error) {
                    this.notificationManager.show('Unable to get your location: ' + error.message, false);
                } else {
                    this.notificationManager.show(message);
                }
            });
        });

        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearAll();
        });

        document.getElementById('request-btn').addEventListener('click', () => {
            this.rideManager.requestRide();
        });

        document.getElementById('cancel-btn').addEventListener('click', () => {
            this.rideManager.cancelRide();
        });

        // Click outside to hide suggestions
        document.addEventListener('click', (e) => {
            if (!originInput.contains(e.target) && !originSuggestions.contains(e.target)) {
                originSuggestions.style.display = 'none';
            }
            if (!destinationInput.contains(e.target) && !destinationSuggestions.contains(e.target)) {
                destinationSuggestions.style.display = 'none';
            }
        });
    }

    clearAll() {
        this.mapManager.clearMarkers();
        document.getElementById('origin-input').value = "";
        document.getElementById('destination-input').value = "";
        document.getElementById('origin-suggestions').style.display = 'none';
        document.getElementById('destination-suggestions').style.display = 'none';
        this.rideDetailsManager.hide();
    }
}

// ============================================================================
// APPLICATION INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', function() {
    const app = new RideApp();
    app.initialize();
});