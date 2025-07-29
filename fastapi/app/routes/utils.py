from shapely.geometry import Point
import geopandas as gpd
from geopy.distance import geodesic
from shapely.geometry import LineString
from shapely.ops import nearest_points
import overpy
import aiohttp
import asyncio
import time
import requests
import math
import random
from app.database.crud import create_location
from app.database.base import SessionLocal
import osmnx as ox
import networkx as nx
from collections import defaultdict
import numpy as np
from itertools import combinations

# data-name="شهرک ریحانه, Mehestan, Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran"
place = "Savojbolagh County, Alborz Province, Iran"
G = ox.graph_from_place(place, network_type='walk')


def get_nearest_nodes_od(df, network_type='walk'):
    """
    Find nearest OSM nodes for origin-destination pairs.

    Args:
        df: DataFrame with 'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng' columns
        network_type: Type of network for OSMNX graph (default: 'walk')

    Returns:
        DataFrame with added 'origin_node' and 'dest_node' columns, and the graph
    """
    # Get bounding box for all points (origins and destinations)
    all_lats = df[['origin_lat', 'destination_lat']].values.flatten()
    all_lons = df[['origin_lng', 'destination_lng']].values.flatten()

    # Use bbox parameter instead of individual north/south/east/west parameters
    # bbox = (36.0087995, 35.9522156, 50.7613994, 50.6636533)

    # G = ox.graph_from_bbox(
    #     bbox=bbox,
    #     network_type=network_type
    # )

    # Find nearest nodes for origins
    df['origin_node'] = ox.distance.nearest_nodes(
        G, X=df['origin_lng'].values, Y=df['origin_lat'].values, return_dist=False
    )

    # Find nearest nodes for destinations
    df['dest_node'] = ox.distance.nearest_nodes(
        G, X=df['destination_lng'].values, Y=df['destination_lat'].values, return_dist=False
    )

    return df, G


def build_distance_matrix(G, nodes, cutoff=1000):
    """
    Build a distance matrix for the given nodes using Dijkstra's algorithm.

    Args:
        G: OSMNX graph
        nodes: List of unique node IDs
        cutoff: Maximum walking distance in meters (default: 1000)

    Returns:
        Dictionary of dictionaries with walking distances between nodes
    """
    return {
        node: nx.single_source_dijkstra_path_length(
            G, node, cutoff=cutoff, weight='length')
        for node in nodes
    }


def calculate_od_pair_similarity(person1, person2, distance_matrix,
                                 origin_weight=0.5, dest_weight=0.5):
    """
    Calculate similarity between two origin-destination pairs.

    Args:
        person1: Dict with 'origin_node' and 'dest_node'
        person2: Dict with 'origin_node' and 'dest_node'
        distance_matrix: Dictionary of walking distances between nodes
        origin_weight: Weight for origin proximity (default: 0.5)
        dest_weight: Weight for destination proximity (default: 0.5)

    Returns:
        Similarity score (lower is more similar)
    """
    # Distance between origins
    origin_dist = distance_matrix.get(person1['origin_node'], {}).get(
        person2['origin_node'], float('inf'))

    # Distance between destinations
    dest_dist = distance_matrix.get(person1['dest_node'], {}).get(
        person2['dest_node'], float('inf'))

    # Combined weighted distance
    if origin_dist == float('inf') or dest_dist == float('inf'):
        return float('inf')

    return origin_weight * origin_dist + dest_weight * dest_dist


def group_people_by_od_similarity(df, distance_matrix, group_size=3,
                                  origin_weight=0.5, dest_weight=0.5, max_distance=1000):
    """
    Group people based on similarity of their origin-destination pairs.

    Args:
        df: DataFrame with origin_node and dest_node columns
        distance_matrix: Dictionary of walking distances between nodes
        group_size: Size of each group (default: 3)
        origin_weight: Weight for origin proximity (default: 0.5)
        dest_weight: Weight for destination proximity (default: 0.5)
        max_distance: Maximum allowed combined distance for grouping

    Returns:
        List of groups, each containing person indices
    """
    people = df.to_dict('records')
    ungrouped = set(range(len(people)))
    groups = []

    while len(ungrouped) >= group_size:
        # Start with an arbitrary person
        current_idx = next(iter(ungrouped))
        current_person = people[current_idx]
        ungrouped.remove(current_idx)

        # Find most similar people
        similarities = []
        for other_idx in ungrouped:
            other_person = people[other_idx]
            similarity = calculate_od_pair_similarity(
                current_person, other_person, distance_matrix,
                origin_weight, dest_weight
            )
            if similarity <= max_distance:
                similarities.append((other_idx, similarity))

        # Sort by similarity and take the closest ones
        similarities.sort(key=lambda x: x[1])

        if len(similarities) >= group_size - 1:
            group = [current_idx]
            for i in range(group_size - 1):
                member_idx = similarities[i][0]
                group.append(member_idx)
                ungrouped.remove(member_idx)
            groups.append(group)
        # If not enough similar people, put the person back for later consideration
        # (This is a simple strategy; more sophisticated approaches could be used)

    return groups


def find_optimal_meeting_points(group_indices, df, distance_matrix, G, all_nodes):
    """
    Find optimal meeting points for origin and destination for a group.

    Args:
        group_indices: List of person indices in the group
        df: DataFrame with origin_node and dest_node columns
        distance_matrix: Dictionary of walking distances between nodes
        G: OSMNX graph
        all_nodes: List of all available nodes to consider as meeting points

    Returns:
        Tuple of ((origin_lat, origin_lng), (destination_lat, destination_lng)) for meeting points
    """
    group_data = df.iloc[group_indices]

    # Find central origin node (can be any node on the map)
    origin_nodes = group_data['origin_node'].tolist()
    central_origin = find_central_node(
        origin_nodes, distance_matrix, all_nodes)

    # Find central destination node (can be any node on the map)
    dest_nodes = group_data['dest_node'].tolist()
    central_dest = find_central_node(dest_nodes, distance_matrix, all_nodes)

    # Convert to coordinates
    origin_coords = None
    dest_coords = None

    if central_origin is not None:
        origin_coords = (G.nodes[central_origin]['y'],
                         G.nodes[central_origin]['x'])

    if central_dest is not None:
        dest_coords = (G.nodes[central_dest]['y'], G.nodes[central_dest]['x'])

    return origin_coords, dest_coords


def find_central_node(group, distance_matrix, all_nodes=None):
    """
    Find the node with minimum total walking distance to others in the group.
    Can consider any node on the map, not just nodes in the group.

    Args:
        group: List of node IDs in a group
        distance_matrix: Dictionary of walking distances between nodes
        all_nodes: List of all available nodes to consider as meeting points.
                  If None, only considers nodes within the group.

    Returns:
        Node ID of the most central node
    """
    if not group:
        return None

    # If all_nodes is provided, consider all nodes as potential meeting points
    candidates = all_nodes if all_nodes is not None else group

    best_node = None
    min_total_distance = float('inf')

    for candidate in candidates:
        # Calculate total distance from this candidate to all group members
        total_distance = 0
        valid_candidate = True

        for group_member in group:
            distance = distance_matrix.get(
                candidate, {}).get(group_member, float('inf'))
            if distance == float('inf'):
                valid_candidate = False
                break
            total_distance += distance

        # Update best candidate if this one is better
        if valid_candidate and total_distance < min_total_distance:
            min_total_distance = total_distance
            best_node = candidate

    return best_node


def get_od_meeting_points(df, network_type='walk', cutoff=1000, group_size=3,
                          origin_weight=0.5, dest_weight=0.5, max_distance=1000):
    """
    Find optimal meeting points for groups of people with similar origin-destination pairs.

    Args:
        df: DataFrame with 'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng' columns
        network_type: Type of network for OSMNX graph (default: 'walk')
        cutoff: Maximum walking distance for distance matrix (default: 1000)
        group_size: Size of each group (default: 3)
        origin_weight: Weight for origin proximity (default: 0.5)
        dest_weight: Weight for destination proximity (default: 0.5)
        max_distance: Maximum allowed combined distance for grouping

    Returns:
        List of tuples: [((origin_lat, origin_lng), (destination_lat, destination_lng)), ...]
        Each tuple contains the meeting points for origins and destinations
    """
    # Step 1: Get nearest OSM nodes for origins and destinations
    df_with_nodes, G = get_nearest_nodes_od(df, network_type)

    # Step 2: Build distance matrix for all unique nodes
    all_nodes = set(df_with_nodes['origin_node'].tolist() +
                    df_with_nodes['dest_node'].tolist())
    distance_matrix = build_distance_matrix(G, list(all_nodes), cutoff)

    # Step 3: Group people by origin-destination similarity
    groups = group_people_by_od_similarity(
        df_with_nodes, distance_matrix, group_size,
        origin_weight, dest_weight, max_distance
    )

    # Step 4: Find optimal meeting points for each group
    meeting_points = []
    for group in groups:
        origin_meeting, dest_meeting = find_optimal_meeting_points(
            group, df_with_nodes, distance_matrix, G, list(all_nodes)
        )
        meeting_points.append((origin_meeting, dest_meeting))

    return meeting_points, groups


# ============================================================================

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

        user_id_start = 10000
        cnt = 0

        # Process all points - snaps to actual road edges, not just intersections
        for item in data:
            origin_random = await get_random_location_in_circle_offline_precise(
                item['origin'][0], item['origin'][1])
            dest_random = await get_random_location_in_circle_offline_precise(
                item['dest'][0], item['dest'][1])

            for i in range(len(dest_random)):
                location_history = create_location(
                    db=db,
                    user_id=user_id_start + cnt,
                    origin_lat=origin_random[i][0],
                    origin_lng=origin_random[i][1],
                    destination_lat=dest_random[i][0],
                    destination_lng=dest_random[i][1]
                )
                cnt += 1
