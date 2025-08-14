import math
import random
from app.database.crud import create_location
from app.database.base import SessionLocal
import osmnx as ox


# data-name="شهرک ریحانه, Mehestan, Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran"
place = "Savojbolagh County, Alborz Province, Iran"
G = ox.graph_from_place(place, network_type='walk')

async def get_random_location_in_circle_offline_precise(center_lat, center_lng, radius_km=1, num_points=150):
    """
    Generate random locations and snap them to actual road segments (not just nodes).
    This is the most accurate version.
    """
    EARTH_RADIUS_KM = 6371.0

    # Download road network once (cached automatically)
    nodes, edges = ox.graph_to_gdfs(G)

    center_lat_rad = math.radians(center_lat)
    center_lng_rad = math.radians(center_lng)

    snapped_points = []

    for _ in range(num_points):
        random_distance = radius_km * math.sqrt(random.random())
        random_bearing = random.random() * 2 * math.pi
        angular_distance = random_distance / EARTH_RADIUS_KM

        new_lat_rad = math.asin(
            math.sin(center_lat_rad) * math.cos(angular_distance) +
            math.cos(center_lat_rad) * math.sin(angular_distance) *
            math.cos(random_bearing)
        )

        new_lng_rad = center_lng_rad + math.atan2(
            math.sin(random_bearing) * math.sin(angular_distance) *
            math.cos(center_lat_rad),
            math.cos(angular_distance) -
            math.sin(center_lat_rad) * math.sin(new_lat_rad)
        )

        new_lat = math.degrees(new_lat_rad)
        new_lng = math.degrees(new_lng_rad)

        # Ensure longitude is within -180 to +180
        new_lng = ((new_lng + 180) % 360) - 180

        # Snap to nearest point on actual road edge (most accurate)
        try:
            snapped_lat, snapped_lng = snap_point_to_road_precise(
                new_lat, new_lng, G)
            snapped_points.append((snapped_lat, snapped_lng))
        except Exception as e:
            print(f"Warning: Could not snap {new_lat}, {new_lng}: {e}")
            snapped_points.append((new_lat, new_lng))

    return snapped_points[0] if num_points == 1 else snapped_points


# Even better approach: Using OSMnx's built-in nearest_edges function
def snap_point_to_road_precise(lat, lng, G):
    """
    Most accurate way to snap to roads using OSMnx's built-in functionality.
    This finds the nearest point on the actual road edge.
    """
    # Find nearest edge (road segment)
    nearest_edge = ox.nearest_edges(G, lng, lat, return_dist=True)

    if len(nearest_edge) == 2:
        edge_info, distance = nearest_edge
        u, v, key = edge_info

        # Get the edge geometry
        edge_data = G.edges[u, v, key]

        if 'geometry' in edge_data:
            # Use the edge geometry
            road_geom = edge_data['geometry']

            from shapely.geometry import Point
            from shapely.ops import nearest_points

            target_point = Point(lng, lat)
            nearest_point_on_road = nearest_points(target_point, road_geom)[1]

            return (nearest_point_on_road.y, nearest_point_on_road.x)
        else:
            # If no geometry, interpolate between nodes
            u_node = G.nodes[u]
            v_node = G.nodes[v]

            # For simplicity, return the closer node
            from geopy.distance import geodesic

            dist_to_u = geodesic((lat, lng), (u_node['y'], u_node['x'])).meters
            dist_to_v = geodesic((lat, lng), (v_node['y'], v_node['x'])).meters

            if dist_to_u < dist_to_v:
                return (u_node['y'], u_node['x'])
            else:
                return (v_node['y'], v_node['x'])
    else:
        # Fallback to nearest node
        nearest_node = ox.nearest_nodes(G, lng, lat)
        node_data = G.nodes[nearest_node]
        return (node_data['y'], node_data['x'])


# Updated main function using offline approach
async def add_random_data():
    """
    Add random data using offline road snapping - much faster and more reliable.
    """
    with SessionLocal() as db:
        data = [
            {
                'origin': [35.97897, 50.73145],
                'dest': [35.962546, 50.678285]
            }
        ]

        origin_points = [
            {'lat': 35.97343762887419, 'lng': 50.733369466165854},
            {'lat': 35.971899682275826, 'lng': 50.73203615202348},
            {'lat': 35.97300620759051, 'lng': 50.73229376277865},
            {'lat': 35.972430973236975, 'lng': 50.730351278776325}
        ]
        dest_points = [
            {'lat': 35.96060642151597, 'lng': 50.67944970979732},
            {'lat': 35.96041031396881, 'lng': 50.67980983794394},
            {'lat': 35.96129782317813, 'lng': 50.68118268477208},
            {'lat': 35.960023756506935, 'lng': 50.68022228130509}
        ]
        user_id_start = 10000
        cnt = 0

        # Process all points - snaps to actual road edges, not just intersections
        for item in data:
            # origin_random = await get_random_location_in_circle_offline_precise(
            #     item['origin'][0], item['origin'][1])
            # dest_random = await get_random_location_in_circle_offline_precise(
            #     item['dest'][0], item['dest'][1])

            for i in range(150):
                # location_history = create_location(
                #     db=db,
                #     user_id=user_id_start + cnt,
                #     origin_lat=origin_random[i][0],
                #     origin_lng=origin_random[i][1],
                #     destination_lat=dest_random[i][0],
                #     destination_lng=dest_random[i][1]
                # )
                cnt += 1
        
        for i in range(len(origin_points)):
            location_history = create_location(
                    db=db,
                    user_id=user_id_start + cnt,
                    origin_lat=origin_points[i]['lat'],
                    origin_lng=origin_points[i]['lng'],
                    destination_lat=dest_points[i]['lat'],
                    destination_lng=dest_points[i]['lng']
                )
            cnt += 1
