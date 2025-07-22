let map = L.map('map').setView([0, 0], 13);
let originMarker = null;
let destinationMarker = null;
let originCoords = null;
let destinationCoords = null;

// Tile layer
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// Try to locate user
map.locate({ setView: true, maxZoom: 16 });

map.on('locationfound', function(e) {
    L.marker(e.latlng).addTo(map).bindPopup("You are here").openPopup();
});

// Map click handler
map.on('click', function(e) {
    if (!originCoords) {
        originCoords = e.latlng;
        originMarker = L.marker(e.latlng, { draggable: true }).addTo(map).bindPopup("Origin").openPopup();
        originMarker.on('dragend', (e) => {
            originCoords = e.target.getLatLng();
            checkBothSelected();
        });
    } else if (!destinationCoords) {
        destinationCoords = e.latlng;
        destinationMarker = L.marker(e.latlng, { draggable: true }).addTo(map).bindPopup("Destination").openPopup();
        destinationMarker.on('dragend', (e) => {
            destinationCoords = e.target.getLatLng();
            checkBothSelected();
        });
    }
    checkBothSelected();
});

function checkBothSelected() {
    if (originCoords && destinationCoords) {
        document.getElementById('request-btn').style.display = 'block';
    }
}

// Request button
document.getElementById('request-btn').addEventListener('click', function () {
    const formData = new FormData();
    formData.append('origin', JSON.stringify(originCoords));
    formData.append('destination', JSON.stringify(destinationCoords));

    fetch('/ride/request/', {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCSRFToken(),
        },
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message || 'Ride Requested');
    })
    .catch(err => {
        console.error(err);
        alert('Failed to request ride.');
    });
});

function getCSRFToken() {
    const name = 'csrftoken';
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let c of cookies) {
            c = c.trim();
            if (c.startsWith(name + '=')) {
                cookieValue = decodeURIComponent(c.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
