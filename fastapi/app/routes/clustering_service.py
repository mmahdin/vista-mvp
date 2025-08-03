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


class SpatialSimilarityMatcher:
    def __init__(self, place: str = "Savojbolagh County, Alborz Province, Iran", 
                 network_type: str = 'walk', k_neighbors: int = 10, 
                 similarity_threshold: float = 0.3, precompute_neighbors: int = 200):
        """
        Initialize the spatial similarity matcher.
        
        Args:
            place: Location to extract graph from
            network_type: Type of network to extract (walk, drive, etc.)
            k_neighbors: Number of closest nodes to consider for each location
            similarity_threshold: Minimum similarity score to form groups
            precompute_neighbors: Number of nearest neighbors to precompute for each node
        """
        self.place = place
        self.network_type = network_type
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.precompute_neighbors = precompute_neighbors
        self.G = None
        self.nodes_gdf = None
        self.node_coords = None
        self.feature_matrix = None
        self.user_locations = None
        self.precomputed_distances = None  # Will store precomputed distances
        self.node_to_index = None  # Mapping from node_id to index in arrays
        
    def _load_graph(self):
        """Load and prepare the OSM graph."""
        print(f"Loading graph for {self.place}...")
        self.G = ox.graph_from_place(self.place, network_type=self.network_type)
        
        # Convert nodes to GeoDataFrame for easier distance calculations
        self.nodes_gdf = ox.graph_to_gdfs(self.G, edges=False)
        
        # Extract node coordinates as numpy array for efficient distance calculation
        self.node_coords = np.array([[row.geometry.y, row.geometry.x] 
                                   for _, row in self.nodes_gdf.iterrows()])
        
        # Create mapping from node_id to index
        self.node_to_index = {node_id: idx for idx, node_id in enumerate(self.nodes_gdf.index)}
        
        print(f"Graph loaded with {len(self.nodes_gdf)} nodes")
    
    def _precompute_distances(self, save_filepath: str = None, load_filepath: str = None):
        """
        Precompute walking distances for each node to its nearest neighbors.
        Can save results to file or load from existing file.
        
        Args:
            save_filepath: Path to save precomputed distances (optional)
            load_filepath: Path to load precomputed distances from (optional)
        """
        # Try to load from file first if specified
        if load_filepath:
            print(f"Attempting to load precomputed distances from {load_filepath}...")
            if self.load_precomputed_distances(load_filepath):
                print("Successfully loaded precomputed distances!")
                return
            else:
                print("Failed to load, will compute distances...")
        
        print(f"Precomputing distances for {self.precompute_neighbors} nearest neighbors per node...")
        
        n_nodes = len(self.nodes_gdf)
        node_list = list(self.nodes_gdf.index)
        
        # Initialize storage for precomputed distances
        # Format: dict[source_node] = [(neighbor_node_idx, distance), ...]
        self.precomputed_distances = {}
        
        # Process each node
        for i, source_node in enumerate(node_list):
            if i % 100 == 0:
                print(f"Precomputing distances for node {i+1}/{n_nodes}")
            
            try:
                # Calculate distances from source node to all reachable nodes
                distances_dict = nx.single_source_dijkstra_path_length(
                    self.G, source_node, weight='length'
                )
                
                # Convert to list of (node_index, distance) pairs
                distance_pairs = []
                for target_node, distance in distances_dict.items():
                    if target_node in self.node_to_index:
                        target_idx = self.node_to_index[target_node]
                        distance_pairs.append((target_idx, distance))
                
                # Sort by distance and keep only the closest neighbors
                distance_pairs.sort(key=lambda x: x[1])
                closest_neighbors = distance_pairs[:self.precompute_neighbors]
                
                # Store in precomputed distances
                source_idx = self.node_to_index[source_node]
                self.precomputed_distances[source_idx] = closest_neighbors
                
            except Exception as e:
                print(f"Warning: Failed to compute distances for node {source_node}: {e}")
                # Store empty list for problematic nodes
                source_idx = self.node_to_index[source_node]
                self.precomputed_distances[source_idx] = []
        
        print("Distance precomputation completed!")
        
        # Save to file if specified
        if save_filepath:
            print(f"Saving precomputed distances to {save_filepath}...")
            self.save_precomputed_distances(save_filepath)
            print("Precomputed distances saved successfully!")
    
    def save_precomputed_distances(self, filepath: str):
        """Save precomputed distances to file for reuse."""
        import pickle
        
        if self.precomputed_distances is None:
            raise ValueError("No precomputed distances to save. Run _precompute_distances first.")
        
        save_data = {
            'precomputed_distances': self.precomputed_distances,
            'node_to_index': self.node_to_index,
            'place': self.place,
            'network_type': self.network_type,
            'precompute_neighbors': self.precompute_neighbors
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Precomputed distances saved to {filepath}")
    
    def load_precomputed_distances(self, filepath: str):
        """Load precomputed distances from file."""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.precomputed_distances = save_data['precomputed_distances']
            self.node_to_index = save_data['node_to_index']
            
            # Verify compatibility
            if (save_data['place'] != self.place or 
                save_data['network_type'] != self.network_type):
                print("Warning: Loaded precomputed distances may not match current graph settings")
            
            print(f"Precomputed distances loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load precomputed distances: {e}")
            return False
    
    def _find_closest_nodes(self, lat: float, lng: float, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k closest nodes to a given coordinate using precomputed walking distances.
        
        Args:
            lat: Latitude
            lng: Longitude
            k: Number of closest nodes (defaults to self.k_neighbors)
            
        Returns:
            Tuple of (node_indices, distances_in_meters)
        """
        if k is None:
            k = self.k_neighbors
        
        # Find the nearest node to the point as starting point
        nearest_node = ox.nearest_nodes(self.G, lng, lat)
        nearest_node_idx = self.node_to_index.get(nearest_node)
        
        if nearest_node_idx is None:
            print(f"Warning: Nearest node not found in index for point ({lat}, {lng})")
            return self._fallback_euclidean_distance(lat, lng, k)
        
        # Get precomputed distances for the nearest node
        if (self.precomputed_distances is None or 
            nearest_node_idx not in self.precomputed_distances):
            print(f"Warning: No precomputed distances for nearest node, using fallback")
            return self._fallback_euclidean_distance(lat, lng, k)

        precomputed_neighbors = self.precomputed_distances[nearest_node_idx]
        
        if len(precomputed_neighbors) == 0:
            print(f"Warning: No precomputed neighbors for nearest node, using fallback")
            return self._fallback_euclidean_distance(lat, lng, k)
        
        # Extract the k closest nodes from precomputed data
        k_actual = min(k, len(precomputed_neighbors))
        closest_k = precomputed_neighbors[:k_actual]
        
        # Separate indices and distances
        closest_indices = np.array([pair[0] for pair in closest_k])
        closest_distances = np.array([pair[1] for pair in closest_k])
        
        return closest_indices, closest_distances
    
    def _fallback_euclidean_distance(self, lat: float, lng: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback method using Euclidean distance when precomputed distances are not available.
        
        Args:
            lat: Latitude
            lng: Longitude
            k: Number of closest nodes
            
        Returns:
            Tuple of (node_indices, distances_in_meters)
        """
        point = np.array([[lat, lng]])
        euclidean_distances = cdist(point, self.node_coords, metric='euclidean')[0]
        # Convert to approximate meters (rough conversion)
        euclidean_distances = euclidean_distances * 111000  # degrees to meters approximation
        
        k_actual = min(k, len(euclidean_distances))
        closest_indices = np.argpartition(euclidean_distances, k_actual)[:k_actual]
        closest_distances = euclidean_distances[closest_indices]
        
        return closest_indices, closest_distances
    
    def _normalize_distances(self, distances: np.ndarray) -> np.ndarray:
        """
        Normalize distances using inverse distance weighting.
        
        Args:
            distances: Array of distances in meters
            
        Returns:
            Normalized weights
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1.0  # 1 meter epsilon for walking distances
        weights = 1.0 / (distances + epsilon)
        
        # Normalize to sum to 1
        return weights / np.sum(weights)
    
    def build_feature_matrix(self, user_locations: List[UserLocation]) -> np.ndarray:
        """
        Build the bipartite feature matrix for all users.
        
        Args:
            user_locations: List of user locations
            
        Returns:
            Feature matrix of shape (n_users, 2*n_nodes)
        """
        if self.G is None:
            self._load_graph()
        
        # Load or compute precomputed distances
        if self.precomputed_distances is None:
            # Create a filename based on place and settings
            filename = '/home/mahdi/Documents/startup/carpooling/v3/vista-mvp/fastapi/prcmd.pkl'
            print("Precomputed distances not found. Computing distances...")
            self._precompute_distances(save_filepath=filename, load_filepath=filename)
            
        self.user_locations = user_locations
        n_users = len(user_locations)
        n_nodes = len(self.nodes_gdf)
        
        # Initialize feature matrix: n_users x (2*n_nodes)
        # First n_nodes columns for origins, next n_nodes for destinations
        feature_matrix = np.zeros((n_users, 2 * n_nodes))
        
        print(f"Building feature matrix for {n_users} users...")
        
        for i, user in enumerate(user_locations):
            if i % 10 == 0:  # Progress indicator
                print(f"Processing user {i+1}/{n_users}")
                
            # Process origin
            origin_indices, origin_distances = self._find_closest_nodes(
                user.origin_lat, user.origin_lng
            )
            origin_weights = self._normalize_distances(origin_distances)
            
            # Place origin weights in first half of columns
            for idx, weight in zip(origin_indices, origin_weights):
                feature_matrix[i, idx] = weight
            
            # Process destination
            dest_indices, dest_distances = self._find_closest_nodes(
                user.destination_lat, user.destination_lng
            )
            dest_weights = self._normalize_distances(dest_distances)
            
            # Place destination weights in second half of columns
            for idx, weight in zip(dest_indices, dest_weights):
                feature_matrix[i, n_nodes + idx] = weight
        
        self.feature_matrix = feature_matrix
        print("Feature matrix built successfully")
        return feature_matrix
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute similarity matrix using inner product (MIPS).
        
        Returns:
            Similarity matrix of shape (n_users, n_users)
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not built. Call build_feature_matrix first.")
        
        # Normalize feature vectors for better similarity computation
        normalized_features = normalize(self.feature_matrix, norm='l2')
        
        # Compute inner product similarity
        similarity_matrix = np.dot(normalized_features, normalized_features.T)
        
        return similarity_matrix
    
    def _calculate_meeting_point(self, locations: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate centroid meeting point for a list of locations.
        
        Args:
            locations: List of (lat, lng) tuples
            
        Returns:
            Centroid coordinates as (lat, lng)
        """
        if not locations:
            return None
            
        avg_lat = sum(loc[0] for loc in locations) / len(locations)
        avg_lng = sum(loc[1] for loc in locations) / len(locations)
        
        return (avg_lat, avg_lng)
    
    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        """
        Find similar user groups using spatial similarity.
        
        Args:
            user_locations: List of user locations
            
        Returns:
            List of ClusterGroup objects
        """
        # Build feature matrix
        self.build_feature_matrix(user_locations)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix()
        
        groups = []
        used_users = set()
        
        print(f"Finding groups with similarity threshold: {self.similarity_threshold}")
        
        for i, user1 in enumerate(user_locations):
            if user1.user_id in used_users:
                continue
                
            # Find most similar users
            similarities = similarity_matrix[i]
            
            # Get indices sorted by similarity (excluding self)
            similar_indices = np.argsort(similarities)[::-1]
            
            group_users = [user1]
            group_similarities = []
            
            for j in similar_indices:
                if j == i:  # Skip self
                    continue
                    
                user2 = user_locations[j]
                similarity_score = similarities[j]
                
                if (user2.user_id not in used_users and 
                    similarity_score >= self.similarity_threshold and 
                    len(group_users) < 3):
                    
                    group_users.append(user2)
                    group_similarities.append(similarity_score)
                    
                    if len(group_users) == 3:
                        break
            
            # Create group if we have at least 2 users with sufficient similarity
            if len(group_users) >= 2:
                # Calculate meeting points
                origin_coords = [user.origin_coords for user in group_users]
                dest_coords = [user.destination_coords for user in group_users]
                
                origin_meeting = self._calculate_meeting_point(origin_coords)
                dest_meeting = self._calculate_meeting_point(dest_coords)
                
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
                
                # Mark users as used
                for user in group_users:
                    used_users.add(user.user_id)
                
                avg_similarity = f"{np.mean(group_similarities):.3f}" if group_similarities else "N/A"
                print(f"Created group {group_id} with {len(group_users)} users (avg similarity: {avg_similarity})")

        
        print(f"Found {len(groups)} groups total")
        return groups
    
    def get_user_recommendations(self, target_user: UserLocation, 
                               all_users: List[UserLocation], 
                               n_recommendations: int = 5) -> List[Tuple[UserLocation, float]]:
        """
        Get top N similar users for a target user.
        
        Args:
            target_user: The user to find recommendations for
            all_users: All available users
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (UserLocation, similarity_score) tuples
        """
        # Build feature matrix including target user
        extended_users = [target_user] + [u for u in all_users if u.user_id != target_user.user_id]
        self.build_feature_matrix(extended_users)
        
        # Compute similarity
        similarity_matrix = self.compute_similarity_matrix()
        
        # Get similarities for target user (first row)
        target_similarities = similarity_matrix[0, 1:]  # Exclude self-similarity
        
        # Get top N recommendations
        top_indices = np.argsort(target_similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            similar_user = all_users[idx]
            similarity_score = target_similarities[idx]
            recommendations.append((similar_user, similarity_score))
        
        return recommendations
    

class ClusteringService:
    """Main clustering service that runs in background threads"""

    def __init__(self, clustering_interval: int = 30, max_wait_time: int = 300):
        self.clustering_interval = clustering_interval
        self.max_wait_time = max_wait_time
        # i need this class
        self.clusterer = SpatialSimilarityMatcher()
        
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
                    new_groups = self.clusterer.cluster_users(available_users)
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
                        
                # Clean up expired groups
                self._cleanup_expired_groups()

            # except Exception as e:
            #     logger.error(f"Error in clustering worker: {e}")

            # Wait for next iteration or stop signal
                self._stop_event.wait(self.clustering_interval)

        logger.info("Clustering worker stopped")

    def _cleanup_expired_groups(self):
        """Clean up expired incomplete groups"""
        current_time = datetime.now(timezone.utc)
        expired_groups = []

        with self._lock:
            for group_id, group in self.active_groups.items():
                if (not group.is_complete() and 
                    (current_time - group.created_at).total_seconds() > self.max_wait_time):
                    expired_groups.append(group_id)

            for group_id in expired_groups:
                group = self.active_groups[group_id]
                for user in group.users:
                    if user.user_id in self.user_to_group:
                        del self.user_to_group[user.user_id]
                del self.active_groups[group_id]

        # Notify observers (outside of lock)
        for group_id in expired_groups:
            self._notify_observers('group_expired', {'group_id': group_id})
            logger.info(f"Expired incomplete group {group_id}")

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