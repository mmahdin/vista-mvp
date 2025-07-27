import osmnx as ox
import networkx as nx
from collections import defaultdict
import numpy as np
from itertools import combinations


def get_nearest_nodes_od(df, network_type='walk'):
    """
    Find nearest OSM nodes for origin-destination pairs.

    Args:
        df: DataFrame with 'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon' columns
        network_type: Type of network for OSMNX graph (default: 'walk')

    Returns:
        DataFrame with added 'origin_node' and 'dest_node' columns, and the graph
    """
    # Get bounding box for all points (origins and destinations)
    all_lats = df[['origin_lat', 'dest_lat']].values.flatten()
    all_lons = df[['origin_lon', 'dest_lon']].values.flatten()

    G = ox.graph_from_bbox(
        north=all_lats.max(),
        south=all_lats.min(),
        east=all_lons.max(),
        west=all_lons.min(),
        network_type=network_type
    )

    # Find nearest nodes for origins
    df['origin_node'] = ox.distance.nearest_nodes(
        G, X=df['origin_lon'].values, Y=df['origin_lat'].values, return_dist=False
    )

    # Find nearest nodes for destinations
    df['dest_node'] = ox.distance.nearest_nodes(
        G, X=df['dest_lon'].values, Y=df['dest_lat'].values, return_dist=False
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
        Tuple of ((origin_lat, origin_lon), (dest_lat, dest_lon)) for meeting points
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
        df: DataFrame with 'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon' columns
        network_type: Type of network for OSMNX graph (default: 'walk')
        cutoff: Maximum walking distance for distance matrix (default: 1000)
        group_size: Size of each group (default: 3)
        origin_weight: Weight for origin proximity (default: 0.5)
        dest_weight: Weight for destination proximity (default: 0.5)
        max_distance: Maximum allowed combined distance for grouping

    Returns:
        List of tuples: [((origin_lat, origin_lon), (dest_lat, dest_lon)), ...]
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


# Example usage for origin-destination clustering:
"""
import pandas as pd

# Create sample data with origin-destination pairs
df_od = pd.DataFrame({
    'person_id': ['A', 'B', 'C', 'D', 'E', 'F'],
    'origin_lat': [40.7589, 40.7614, 40.7505, 40.7580, 40.7600, 40.7520],
    'origin_lon': [-73.9851, -73.9776, -73.9934, -73.9855, -73.9800, -73.9900],
    'dest_lat': [40.7489, 40.7514, 40.7405, 40.7480, 40.7500, 40.7420],
    'dest_lon': [-73.9951, -73.9876, -74.0034, -73.9955, -73.9900, -74.0000]
})

# Get meeting points for origin-destination pairs
meeting_points, groups = get_od_meeting_points(
    df_od, 
    group_size=3,
    origin_weight=0.6,  # Prioritize origin proximity slightly more
    dest_weight=0.4,    # Destination proximity has less weight
    max_distance=800    # Maximum combined distance for grouping
)

print("Groups formed:", groups)
for i, (origin_meeting, dest_meeting) in enumerate(meeting_points):
    print(f"Group {i+1}:")
    print(f"  Origin meeting point: {origin_meeting}")
    print(f"  Destination meeting point: {dest_meeting}")
"""
