# clustering_service.py
import asyncio
import logging
import threading
from typing import List, Dict, Optional, Tuple, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import numpy as np
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import time
from sklearn.neighbors import BallTree
import pickle
import itertools
import time
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import linear_kernel
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
from .connection_manager import connection_manager 
import math

from app.database import get_db, SessionLocal
from app.database.models import *


place = "Savojbolagh County, Alborz Province, Iran"
G = ox.graph_from_place(place, network_type='walk')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserLocation:
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    stored_at: datetime

    @property
    def origin_coords(self) -> Tuple[float, float]:
        return (self.origin_lat, self.origin_lng)

    @property
    def destination_coords(self) -> Tuple[float, float]:
        return (self.destination_lat, self.destination_lng)

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'origin_lat': self.origin_lat,
            'origin_lng': self.origin_lng,
            'destination_lat': self.destination_lat,
            'destination_lng': self.destination_lng,
            'stored_at': self.stored_at.isoformat()
        }


@dataclass
class ClusterGroup:
    group_id: str
    users: List[UserLocation]
    created_at: datetime
    meeting_point_origin: Optional[Tuple[float, float]] = None
    meeting_point_destination: Optional[Tuple[float, float]] = None
    status: str = "forming"  # forming, complete, expired

    def is_complete(self) -> bool:
        return len(self.users) == 3

    def has_user(self, user_id: int) -> bool:
        return any(u.user_id == user_id for u in self.users)

    def get_user_ids(self) -> List[int]:
        return [u.user_id for u in self.users]

    def to_dict(self) -> Dict:
        return {
            'group_id': self.group_id,
            'users': [user.to_dict() for user in self.users],
            'created_at': self.created_at.isoformat(),
            'meeting_point_origin': self.meeting_point_origin,
            'meeting_point_destination': self.meeting_point_destination,
            'status': self.status,
            'user_count': len(self.users)
        }


class SpatialClusteringSystem:
    def __init__(self, place: str = "Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran", 
                 k_nearest: int = 100, similarity_threshold: float = 0.7,
                 cache_file: str = "spatial_cache.pkl"):
        """
        Initialize the spatial clustering system.
        
        Args:
            place: Location to extract graph from OSM
            k_nearest: Number of nearest nodes to consider for each location
            similarity_threshold: Minimum similarity threshold for clustering
            cache_file: File to cache precomputed data
        """
        self.place = place
        self.k_nearest = k_nearest
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        
        # Graph and precomputed data
        self.G = None
        self.nodes_list = None
        self.node_to_idx = None
        self.nearest_nodes_cache = {}
        self.distance_cache = {}
        
        # BallTree for fast nearest neighbor search
        self.ball_tree = None
        self.node_coords = None
        
        # Initialize the system
        self._load_or_compute_graph()
        self._build_ball_tree()
        self._load_or_compute_precomputed_data()
    
    def _load_or_compute_graph(self):
        """Load OSM graph for the specified place."""
        print(f"Loading graph for {self.place}...")
        self.G = ox.graph_from_place(self.place, network_type='walk')
        self.nodes_list = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        print(f"Graph loaded with {len(self.nodes_list)} nodes")
    
    def _build_ball_tree(self):
        """Build BallTree for fast nearest neighbor search."""
        print("Building BallTree for fast nearest neighbor search...")
        
        # Extract coordinates for all nodes in radians (required for haversine distance)
        self.node_coords = []
        for node in self.nodes_list:
            lat = np.radians(self.G.nodes[node]['y'])
            lng = np.radians(self.G.nodes[node]['x'])
            self.node_coords.append([lat, lng])
        
        self.node_coords = np.array(self.node_coords)
        
        # Build BallTree with haversine metric (great for geographic coordinates)
        self.ball_tree = BallTree(self.node_coords, metric='haversine')
        print("BallTree built successfully")
    
    def _load_or_compute_precomputed_data(self):
        """Load precomputed data or compute if doesn't exist."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.nearest_nodes_cache = cache_data['nearest_nodes']
                self.distance_cache = cache_data['distances']
                print("Loaded precomputed data from cache")
        except FileNotFoundError:
            print("Cache not found. Computing precomputed data...")
            self._precompute_data()
            self._save_cache()
    
    def _precompute_data(self):
        """Precompute nearest nodes and distances for optimization."""
        print("Precomputing nearest nodes and distances...")
        
        # For each node, find 200 nearest nodes and their distances
        for i, node in enumerate(self.nodes_list):
            if i % 100 == 0:
                print(f"Processing node {i}/{len(self.nodes_list)}")
            
            try:
                # Get distances to all other nodes (limited by network connectivity)
                distances = nx.single_source_dijkstra_path_length(
                    self.G, node, cutoff=5000, weight='length'  # 5km cutoff
                )
                
                # Sort by distance and take top 200
                sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:200]
                
                # Store nearest nodes
                nearest_nodes = [(target_node, dist) for target_node, dist in sorted_distances]
                self.nearest_nodes_cache[node] = nearest_nodes
                
                # Store distance matrix for this node
                self.distance_cache[node] = {target: dist for target, dist in sorted_distances}
                
            except Exception as e:
                print(f"Error processing node {node}: {e}")
                # Fallback to Euclidean distance
                node_coords = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                euclidean_distances = []
                
                for other_node in self.nodes_list:
                    if other_node != node:
                        other_coords = (self.G.nodes[other_node]['y'], self.G.nodes[other_node]['x'])
                        dist = ox.distance.euclidean_dist_vec(
                            node_coords[0], node_coords[1], 
                            other_coords[0], other_coords[1]
                        )
                        euclidean_distances.append((other_node, dist))
                
                euclidean_distances.sort(key=lambda x: x[1])
                self.nearest_nodes_cache[node] = euclidean_distances[:200]
                self.distance_cache[node] = {target: dist for target, dist in euclidean_distances[:200]}
    
    def _save_cache(self):
        """Save precomputed data to cache file."""
        cache_data = {
            'nearest_nodes': self.nearest_nodes_cache,
            'distances': self.distance_cache
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Precomputed data saved to cache")
    
    def _find_k_nearest_nodes(self, coords: Tuple[float, float]) -> List[Tuple[int, float]]:
        """Find k nearest nodes to given coordinates using BallTree."""
        lat, lng = coords
        
        # Convert coordinates to radians for BallTree query
        query_point = np.array([[np.radians(lat), np.radians(lng)]])
        
        # Find nearest node using BallTree (much faster than ox.distance.nearest_nodes)
        distances, indices = self.ball_tree.query(query_point, k=1)
        nearest_node_idx = indices[0][0]
        nearest_node = self.nodes_list[nearest_node_idx]
        
        # Get precomputed nearest nodes
        if nearest_node in self.nearest_nodes_cache:
            candidates = self.nearest_nodes_cache[nearest_node]
        else:
            # Fallback to BallTree for finding k nearest nodes directly
            distances, indices = self.ball_tree.query(query_point, k=min(self.k_nearest, len(self.nodes_list)))
            candidates = []
            
            for i, idx in enumerate(indices[0]):
                node = self.nodes_list[idx]
                # Convert distance from radians back to meters (approximate)
                dist_meters = distances[0][i] * 6371000  # Earth's radius in meters
                candidates.append((node, dist_meters))
        
        # Return top k candidates
        return candidates[:self.k_nearest]
    
    def _normalize_distances(self, distances: List[float], sigma: float = 500.0) -> List[float]:
        """Convert distances to similarity weights using a Gaussian kernel.
        
        Args:
            distances: list of distances in meters
            sigma: scale parameter controlling decay rate (meters)
        """
        if not distances:
            return []

        # Gaussian kernel: sim = exp(-(d^2) / (2*sigma^2))
        similarities = [math.exp(-(d ** 2) / (2 * sigma ** 2)) for d in distances]

        # Keep magnitudes — no sum-to-1 normalization
        return similarities
    
    def _create_feature_matrix(self, user_locations: List[UserLocation]) -> np.ndarray:
        """Create the bipartite feature matrix for users."""
        n_nodes = len(self.nodes_list)
        n_users = len(user_locations)
        
        # Matrix with 2n columns (n for origins, n for destinations)
        matrix = np.zeros((n_users, 2 * n_nodes))

        start_time = time.time()
        origin_nearest = self._find_k_nearest_nodes(user_locations[0].origin_coords)
        end_time = time.time()
        print(f"Time taken for _find_k_nearest_nodes: {end_time - start_time:.4f} seconds")
        
        for user_idx, user in enumerate(user_locations):
            # Process origin
            origin_nearest = self._find_k_nearest_nodes(user.origin_coords)
            origin_distances = [dist for _, dist in origin_nearest]
            origin_weights = self._normalize_distances(origin_distances)
            
            for (node, _), weight in zip(origin_nearest, origin_weights):
                node_idx = self.node_to_idx[node]
                matrix[user_idx, node_idx] = weight  # Origin columns

            
            # Process destination
            dest_nearest = self._find_k_nearest_nodes(user.destination_coords)
            dest_distances = [dist for _, dist in dest_nearest]
            dest_weights = self._normalize_distances(dest_distances)
            
            for (node, _), weight in zip(dest_nearest, dest_weights):
                node_idx = self.node_to_idx[node]
                matrix[user_idx, n_nodes + node_idx] = weight  # Destination columns

        return matrix
    
    def _calculate_meeting_points(self, users: List[UserLocation]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Calculate meeting points for origin and destination."""
        if not users:
            return None, None
        
        # Calculate centroid for origins
        origin_lats = [u.origin_lat for u in users]
        origin_lngs = [u.origin_lng for u in users]
        origin_meeting = (np.mean(origin_lats), np.mean(origin_lngs))
        
        # Calculate centroid for destinations
        dest_lats = [u.destination_lat for u in users]
        dest_lngs = [u.destination_lng for u in users]
        dest_meeting = (np.mean(dest_lats), np.mean(dest_lngs))
        
        return origin_meeting, dest_meeting
    
    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        """
        Main clustering function that groups similar users.
        
        Args:
            user_locations: List of UserLocation objects
            
        Returns:
            List of ClusterGroup objects
        """
        if len(user_locations) < 2:
            return []
        
        print(f"Clustering {len(user_locations)} users...")
        
        # Create feature matrix
        
        start_time = time.time()
        feature_matrix = self._create_feature_matrix(user_locations)
        end_time = time.time()
        print(f"Time taken for feature_matrix: {end_time - start_time:.4f} seconds")
        
        # Calculate similarity matrix using cosine similarity (approximates MIPS for normalized vectors)
        similarity_matrix = cosine_similarity(feature_matrix)

        # Find groups
        groups = []
        used_users = set()
        
        start_time = time.time()
        for i, user1 in enumerate(user_locations):
            if user1.user_id in used_users:
                continue
            
            # Find most similar users
            similarities = similarity_matrix[i]
            similar_indices = []
            
            for j, sim in enumerate(similarities):
                if i != j and sim >= self.similarity_threshold and user_locations[j].user_id not in used_users:
                    similar_indices.append((j, sim))
            
            # Sort by similarity
            similar_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Form groups of up to 3 users
            if similar_indices:
                group_users = [user1]
                used_users.add(user1.user_id)
                
                # Add up to 2 more users
                for j, sim in similar_indices[:2]:
                    if user_locations[j].user_id not in used_users:
                        group_users.append(user_locations[j])
                        used_users.add(user_locations[j].user_id)
                
                # Calculate meeting points
                origin_meeting, dest_meeting = self._calculate_meeting_points(group_users)
                
                # Create group
                group_id = f"group_{'_'.join(str(u.user_id) for u in group_users)}_{int(time.time())}"
                group = ClusterGroup(
                    group_id=group_id,
                    users=group_users,
                    created_at=datetime.now(timezone.utc),
                    meeting_point_origin=origin_meeting,
                    meeting_point_destination=dest_meeting,
                    status="complete" if len(group_users) == 3 else "forming"
                )
                groups.append(group)
        end_time = time.time()
        print(f"Time taken for gouping: {end_time - start_time:.4f} seconds")
        print(f"Created {len(groups)} groups")
        return groups


class ClusteringService:
    """Main clustering service that runs in background threads"""

    def __init__(self, clustering_interval: int = 5, max_wait_time: int = 300):
        self.clustering_interval = clustering_interval
        self.max_wait_time = max_wait_time
        # i need this class
        self.clusterer = SpatialClusteringSystem()
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        self.active_groups: Dict[str, ClusterGroup] = {}
        self.user_to_group: Dict[int, str] = {}
        
        # Threading
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clustering")
        
        # Observers
        self._observers: List[Callable] = []

    def add_observer(self, callback: Callable):
        """Add observer for group updates"""
        with self._lock:
            self._observers.append(callback)

    def remove_observer(self, callback: Callable):
        """Remove observer"""
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def _notify_observers(self, event_type: str, data: dict):
        """Notify all observers of changes"""
        print(data['users'])
        with self._lock:
            observers = self._observers.copy()
        
        for callback in observers:
            try:
                # Schedule the callback to run in the executor
                self._executor.submit(callback, event_type, data)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

    def _get_recent_locations(self) -> List[UserLocation]:
        """Get all location requests from database"""
        
        db = SessionLocal()
        try:
            locations = db.query(Location).all()

            return [
                UserLocation(
                    user_id=loc.user_id,
                    origin_lat=loc.origin_lat,
                    origin_lng=loc.origin_lng,
                    destination_lat=loc.destination_lat,
                    destination_lng=loc.destination_lng,
                    stored_at=loc.stored_at
                )
                for loc in locations
            ]
        finally:
            db.close()

    def _clustering_worker(self):
        """Main clustering worker that runs in background thread"""
        logger.info("Clustering worker started")
        
        while not self._stop_event.is_set():
            # try:
                # Get recent locations
                recent_locations = self._get_recent_locations()

                # Filter out users who are already in complete groups
                with self._lock:
                    available_users = [
                        user for user in recent_locations
                        if user.user_id not in self.user_to_group or
                        not self.active_groups.get(self.user_to_group[user.user_id], ClusterGroup("", [], datetime.now())).is_complete()
                    ]

                if len(available_users) >= 3:
                    # Perform clustering
                    start_time = time.time()
                    new_groups = self.clusterer.cluster_users(recent_locations)
                    end_time = time.time()
                    print(f"Time taken: {end_time - start_time:.4f} seconds")

                    # Process new groups
                    with self._lock:
                        for group in new_groups:
                            self.active_groups[group.group_id] = group
                            
                            # Update user mappings
                            for user in group.users:
                                self.user_to_group[user.user_id] = group.group_id

                    # Notify observers (outside of lock)
                    for group in new_groups:
                        self._notify_observers('group_formed', {
                            'group_id': group.group_id,
                            'users': group.get_user_ids(),
                            'group_data': group.to_dict()
                        })

                        self._notify_group_via_websocket(group)

            # except Exception as e:
            #     logger.error(f"Error in clustering worker: {e}")

            # Wait for next iteration or stop signal
                self._stop_event.wait(self.clustering_interval)

        logger.info("Clustering worker stopped")

    def _notify_group_via_websocket(self, group: ClusterGroup):
        """Send WebSocket notifications to all users in a group"""
        
        for user in group.users:
            # Get companions for this user (excluding themselves)
            companions = [u for u in group.users if u.user_id != user.user_id]
            
            message = {
                'type': 'group_formed',
                'group_id': group.group_id,
                'companions': [
                    {
                        'user_id': comp.user_id,
                        'origin_lat': comp.origin_lat,
                        'origin_lng': comp.origin_lng,
                        'destination_lat': comp.destination_lat,
                        'destination_lng': comp.destination_lng,
                        'stored_at': comp.stored_at.isoformat()
                    }
                    for comp in companions
                ],
                'meeting_point_origin': group.meeting_point_origin,
                'meeting_point_destination': group.meeting_point_destination,
                'created_at': group.created_at.isoformat()
            }
            
            # Schedule WebSocket notification
            self._executor.submit(self._send_websocket_message, user.user_id, message)

    def _send_websocket_message(self, user_id: int, message: dict):
        """Helper method to send WebSocket message"""
        try:            
            # Create new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Send the message
            loop.run_until_complete(connection_manager.send_group_update(user_id, message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message to user {user_id}: {e}")

    def start(self):
        """Start the clustering service"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Clustering service is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._clustering_worker, daemon=True)
        self._thread.start()
        logger.info("Clustering service started")

    def stop(self):
        """Stop the clustering service"""
        if self._thread is None:
            return

        logger.info("Stopping clustering service...")
        self._stop_event.set()
        
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        
        self._executor.shutdown(wait=False)
        logger.info("Clustering service stopped")

    # Public API methods for FastAPI endpoints    
    def get_user_group(self, user_id: int) -> Optional[Dict]:
        """Get group information for a specific user"""
        with self._lock:
            if user_id not in self.user_to_group:
                return None
                
            group_id = self.user_to_group[user_id]
            group = self.active_groups.get(group_id)
            
            if not group:
                return None
                
            return group.to_dict()

    def get_user_companions(self, user_id: int) -> Optional[Dict]:
        """Get companions for a specific user"""
        with self._lock:
            if user_id not in self.user_to_group:
                return None

            group_id = self.user_to_group[user_id]
            group = self.active_groups.get(group_id)

            # if not group or not group.is_complete():
            #     return None

            companions = [u for u in group.users if u.user_id != user_id]
            return {
                'group_id': group_id,
                'companions': [comp.to_dict() for comp in companions],
                'meeting_point_origin': group.meeting_point_origin,
                'meeting_point_destination': group.meeting_point_destination,
                'created_at': group.created_at.isoformat()
            }

    def get_user_meeting_points(self, user_id: int) -> Optional[Dict]:
        """Get meeting points for a user's group"""
        with self._lock:
            if user_id not in self.user_to_group:
                return None

            group_id = self.user_to_group[user_id]
            group = self.active_groups.get(group_id)

            if not group:
                return None

            return {
                'group_id': group_id,
                'meeting_point_origin': group.meeting_point_origin,
                'meeting_point_destination': group.meeting_point_destination,
                'status': group.status
            }

    def get_all_active_groups(self) -> List[Dict]:
        """Get all active groups"""
        with self._lock:
            return [group.to_dict() for group in self.active_groups.values()]

    def get_service_status(self) -> Dict:
        """Get current service status"""
        with self._lock:
            return {
                'is_running': self._thread is not None and self._thread.is_alive(),
                'active_groups': len(self.active_groups),
                'complete_groups': len([g for g in self.active_groups.values() if g.is_complete()]),
                'users_in_groups': len(self.user_to_group),
                'clustering_interval': self.clustering_interval,
                'max_wait_time': self.max_wait_time
            }

    def remove_user_from_group(self, user_id: int) -> bool:
        """Remove a user from their current group"""
        with self._lock:
            if user_id not in self.user_to_group:
                return False

            group_id = self.user_to_group[user_id]
            group = self.active_groups.get(group_id)
            
            if not group:
                return False

            # Remove user from group
            group.users = [u for u in group.users if u.user_id != user_id]
            del self.user_to_group[user_id]

            # If group becomes empty, remove it
            if not group.users:
                del self.active_groups[group_id]
                self._notify_observers('group_disbanded', {'group_id': group_id})
            else:
                # Update group status
                group.status = "forming" if not group.is_complete() else "complete"
                self._notify_observers('group_updated', {
                    'group_id': group_id,
                    'group_data': group.to_dict()
                })

            return True


# Global service instance
_clustering_service: ClusteringService | None = None


def get_clustering_service() -> ClusteringService:
    """Get the global clustering service instance"""
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService()
    return _clustering_service


def start_clustering_service():
    """Start the clustering service (call this in FastAPI lifespan)"""
    service = get_clustering_service()
    service.start()


def stop_clustering_service():
    """Stop the clustering service (call this in FastAPI lifespan)"""
    global _clustering_service
    if _clustering_service is not None:
        _clustering_service.stop()
        _clustering_service = None