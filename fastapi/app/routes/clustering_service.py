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


@dataclass
class User:
    id: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float


@dataclass
class Bucket:
    id: int
    center_lat: float
    center_lon: float
    node_count: int
    nodes: List[int]
    bbox: Tuple[float, float, float, float] 


class FastBucketAssigner:
    def __init__(self, buckets: List[Bucket]):
        self.buckets = buckets
        self.num_buckets = len(buckets)
        
        # Create spatial index for fast lookup
        self.bucket_ids = np.array([b.id for b in buckets])
        self.min_lats = np.array([b.bbox[0] for b in buckets])
        self.min_lons = np.array([b.bbox[1] for b in buckets])
        self.max_lats = np.array([b.bbox[2] for b in buckets])
        self.max_lons = np.array([b.bbox[3] for b in buckets])
        
        # Create a mapping from bucket_id to index for quick lookup
        self.bucket_id_to_index = {bucket.id: i for i, bucket in enumerate(buckets)}

    def get_nearest_node(self, lat, lon):
        """
        Given latitude and longitude, return the (lat, lon) of the nearest node in the graph G.
        """
        # Find the nearest node ID
        nearest_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        
        # Extract node coordinates
        node_lat = G.nodes[nearest_node]['y']
        node_lon = G.nodes[nearest_node]['x']
        
        return node_lat, node_lon
        
    def find_bucket_for_point(self, lat: float, lon: float) -> int:
        """Find which bucket a point (lat, lon) belongs to. Returns bucket_id or -1 if not found."""
        lat, lon = self.get_nearest_node(lat, lon)

        inside_mask = (
            (self.min_lats <= lat) & (lat <= self.max_lats) &
            (self.min_lons <= lon) & (lon <= self.max_lons)
        )
        
        indices = np.where(inside_mask)[0]
        if len(indices) > 0:
            return self.bucket_ids[indices[0]]
        return -1
    
    def assign_users_to_od_buckets(self, users: List[UserLocation]) -> Dict[Tuple[int, int], List[UserLocation]]:
        """Assign users to origin-destination bucket pairs."""
        od_buckets = defaultdict(list)
        
        for user in users:
            origin_bucket = self.find_bucket_for_point(user.origin_lat, user.origin_lng)
            dest_bucket = self.find_bucket_for_point(user.destination_lat, user.destination_lng)
            
            if origin_bucket != -1 and dest_bucket != -1:
                od_buckets[(origin_bucket, dest_bucket)].append(user)
        
        return dict(od_buckets)


class PreciseUserGrouper:
    def __init__(self, max_walking_distance_km: float = 0.5):
        """
        Initialize the grouper.
        
        Args:
            max_walking_distance_km: Maximum acceptable walking distance between users in km
        """
        self.max_walking_distance_km = max_walking_distance_km
        self.buckets_file_path = '/home/mahdi/Documents/startup/carpooling/v3/vista-mvp/fastapi/buckets1_data.pkl'
        
        # Load buckets
        self._load_buckets()
        
        # Initialize bucket assigner
        self.bucket_assigner = FastBucketAssigner(self.kmeans_buckets)
        
        logger.info(f"EfficientRideShareClusterer initialized with {len(self.kmeans_buckets)} buckets")
    
    def _load_buckets(self):
        """Load bucket data from pickle file"""
        try:
            with open(self.buckets_file_path, "rb") as f:
                buckets_data = pickle.load(f)
            self.kmeans_buckets = [Bucket(**b) for b in buckets_data]
            logger.info(f"Successfully loaded {len(self.kmeans_buckets)} buckets from {self.buckets_file_path}")
        except FileNotFoundError:
            logger.error(f"Bucket file not found: {self.buckets_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading bucket file: {e}")
            raise
    
    def calculate_walking_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate walking distance between two points in kilometers."""
        return geodesic(point1, point2).kilometers
    
    def calculate_user_distance(self, user1: UserLocation, user2: UserLocation) -> float:
        """
        Calculate the total walking distance between two users.
        This considers both origin-to-origin and destination-to-destination distances.
        """
        origin_distance = self.calculate_walking_distance(
            (user1.origin_lat, user1.origin_lng),
            (user2.origin_lat, user2.origin_lng)
        )
        dest_distance = self.calculate_walking_distance(
            (user1.destination_lat, user1.destination_lng),
            (user2.destination_lat, user2.destination_lng)
        )
        # Return average of origin and destination distances
        return (origin_distance + dest_distance) / 2
    
    def calculate_group_cohesion(self, users: List[UserLocation]) -> float:
        """
        Calculate the cohesion score for a group of users.
        Lower score means users are closer to each other.
        """
        if len(users) < 2:
            return 0.0
        
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                total_distance += self.calculate_user_distance(users[i], users[j])
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def calculate_meeting_points(self, users: List[UserLocation]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate optimal meeting points for origin and destination."""
        if not users:
            return (0.0, 0.0), (0.0, 0.0)
        
        # Calculate centroid for origins
        origin_lats = [u.origin_lat for u in users]
        origin_lngs = [u.origin_lng for u in users]
        origin_meeting = (
            sum(origin_lats) / len(origin_lats),
            sum(origin_lngs) / len(origin_lngs)
        )
        
        # Calculate centroid for destinations
        dest_lats = [u.destination_lat for u in users]
        dest_lngs = [u.destination_lng for u in users]
        dest_meeting = (
            sum(dest_lats) / len(dest_lats),
            sum(dest_lngs) / len(dest_lngs)
        )
        
        return origin_meeting, dest_meeting
    
    def find_optimal_groups_greedy(self, users: List[UserLocation]) -> List[List[UserLocation]]:
        """
        Find optimal groups using a greedy approach.
        This is more efficient for larger user sets.
        """
        if len(users) <= 3:
            return [users] if users else []
        
        remaining_users = users.copy()
        groups = []
        
        while len(remaining_users) >= 3:
            best_group = None
            best_cohesion = float('inf')
            best_indices = None
            
            # Try all combinations of 3 users from remaining users
            for i in range(len(remaining_users)):
                for j in range(i + 1, len(remaining_users)):
                    for k in range(j + 1, len(remaining_users)):
                        candidate_group = [remaining_users[i], remaining_users[j], remaining_users[k]]
                        cohesion = self.calculate_group_cohesion(candidate_group)
                        
                        if cohesion < best_cohesion:
                            best_cohesion = cohesion
                            best_group = candidate_group
                            best_indices = [i, j, k]
            
            if best_group and best_cohesion <= self.max_walking_distance_km:
                groups.append(best_group)
                # Remove users from remaining list (in reverse order to maintain indices)
                for idx in sorted(best_indices, reverse=True):
                    remaining_users.pop(idx)
            else:
                # If no good group found, break to avoid infinite loop
                break
        
        # Handle remaining users (less than 3 or couldn't form good groups)
        if remaining_users:
            if len(remaining_users) >= 2:
                groups.append(remaining_users)
            else:
                # Single user - add to last group if it has space, otherwise create new group
                if groups and len(groups[-1]) < 3:
                    groups[-1].extend(remaining_users)
                else:
                    groups.append(remaining_users)
        
        return groups
    
    def find_optimal_groups_exhaustive(self, users: List[UserLocation]) -> List[List[UserLocation]]:
        """
        Find optimal groups using exhaustive search.
        Use this for smaller user sets (< 10 users) for best results.
        """
        if len(users) <= 3:
            return [users] if users else []
        
        n = len(users)
        best_groups = None
        best_total_cohesion = float('inf')
        
        # Generate all possible ways to partition users into groups of 3
        for partition in self._generate_partitions(list(range(n)), 3):
            total_cohesion = 0.0
            valid_partition = True
            
            for group_indices in partition:
                if len(group_indices) < 2:
                    continue
                    
                group_users = [users[i] for i in group_indices]
                cohesion = self.calculate_group_cohesion(group_users)
                
                if len(group_indices) == 3 and cohesion > self.max_walking_distance_km:
                    valid_partition = False
                    break
                
                total_cohesion += cohesion
            
            if valid_partition and total_cohesion < best_total_cohesion:
                best_total_cohesion = total_cohesion
                best_groups = [[users[i] for i in group_indices] for group_indices in partition]
        
        return best_groups or [users]
    
    def _generate_partitions(self, items: List[int], group_size: int) -> List[List[List[int]]]:
        """Generate all possible partitions of items into groups of specified size."""
        if not items:
            return [[]]
        
        if len(items) < group_size:
            return [[items]]
        
        partitions = []
        
        # Try to form a group of the specified size
        for combo in itertools.combinations(items, group_size):
            remaining = [x for x in items if x not in combo]
            for sub_partition in self._generate_partitions(remaining, group_size):
                partitions.append([list(combo)] + sub_partition)
        
        # Also try with remaining items as a single group (if not empty)
        if len(items) != group_size:
            partitions.append([items])
        
        return partitions
    
    def cluster_users(self, available_users: List[UserLocation]) -> List[ClusterGroup]:
        """
        Main method to create cluster groups from OD assignments.
        """
        od_assignments = self.bucket_assigner.assign_users_to_od_buckets(available_users)

        groups = []
        
        for (origin_bucket, dest_bucket), users in od_assignments.items():
            if not users:
                continue
            
            # Choose algorithm based on user count
            if len(users) <= 9:  # Use exhaustive for small groups
                optimal_groups = self.find_optimal_groups_exhaustive(users)
            else:  # Use greedy for larger groups
                optimal_groups = self.find_optimal_groups_greedy(users)
            
            # Create ClusterGroup objects
            for group_users in optimal_groups:
                if not group_users:
                    continue
                
                # Calculate meeting points
                origin_meeting, dest_meeting = self.calculate_meeting_points(group_users)
                
                # Generate group ID
                user_ids = sorted([u.user_id for u in group_users])
                group_id = f"group_{'_'.join(map(str, user_ids))}_{int(time.time())}"
                
                # Determine status
                status = "complete" if len(group_users) == 3 else "forming"
                
                group = ClusterGroup(
                    group_id=group_id,
                    users=group_users,
                    created_at=datetime.now(timezone.utc),
                    meeting_point_origin=origin_meeting,
                    meeting_point_destination=dest_meeting,
                    status=status
                )
                
                groups.append(group)
        
        return groups
    

class ClusteringService:
    """Main clustering service that runs in background threads"""

    def __init__(self, clustering_interval: int = 30, max_wait_time: int = 300):
        self.clustering_interval = clustering_interval
        self.max_wait_time = max_wait_time
        # i need this class
        self.clusterer = PreciseUserGrouper()
        
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
            try:
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

            except Exception as e:
                logger.error(f"Error in clustering worker: {e}")

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

            if not group or not group.is_complete():
                return None

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