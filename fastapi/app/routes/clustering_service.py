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

from app.database import get_db, SessionLocal
from app.database.models import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Precompute the OSM graph for the entire area
PLACE = "Savojbolagh County, Alborz Province, Iran"
GLOBAL_GRAPH = ox.graph_from_place(PLACE, network_type='walk', simplify=True)
NODE_COORDS = np.array([(data['y'], data['x']) for node, data in GLOBAL_GRAPH.nodes(data=True)])
NODE_COORDS_RAD = np.radians(NODE_COORDS)
NODE_TREE = BallTree(NODE_COORDS_RAD, metric='haversine')


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





class ClusteringService:
    """Main clustering service that runs in background threads"""

    def __init__(self, clustering_interval: int = 30, max_wait_time: int = 300):
        self.clustering_interval = clustering_interval
        self.max_wait_time = max_wait_time
        # i need this class
        self.clusterer = EfficientRideShareClusterer()
        
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
                    new_groups = self.clusterer.cluster_users(available_users)

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
                        
                        logger.info(f"Formed new group {group.group_id} with users: {group.get_user_ids()}")

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