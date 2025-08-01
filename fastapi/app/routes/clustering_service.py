# clustering_service.py
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import numpy as np
from geopy.distance import geodesic
import osmnx as ox
import networkx as nx
from collections import defaultdict
import json

from app.database import get_db, SessionLocal
from app.database.models import *

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


@dataclass
class UserGroup:
    users: List[UserLocation]
    created_at: datetime
    group_id: str

    def is_complete(self) -> bool:
        return len(self.users) == 3

    def has_user(self, user_id: int) -> bool:
        return any(u.user_id == user_id for u in self.users)


class OSMDistanceCalculator:
    """Handles walking distance calculations using OSM data"""

    def __init__(self, cache_size: int = 1000):
        self.graph_cache = {}
        self.distance_cache = {}
        self.cache_size = cache_size

    def _get_or_create_graph(self, center_lat: float, center_lng: float, radius: int = 2000) -> nx.Graph:
        """Get or create OSM walking graph for the area"""
        cache_key = f"{center_lat:.4f},{center_lng:.4f},{radius}"

        if cache_key not in self.graph_cache:
            try:
                # Download walking network around the center point
                G = ox.graph_from_point(
                    (center_lat, center_lng),
                    dist=radius,
                    network_type='walk',
                    simplify=True
                )
                self.graph_cache[cache_key] = G

                # Limit cache size
                if len(self.graph_cache) > self.cache_size:
                    oldest_key = next(iter(self.graph_cache))
                    del self.graph_cache[oldest_key]

            except Exception as e:
                logger.warning(f"Failed to get OSM graph: {e}")
                return None

        return self.graph_cache.get(cache_key)

    def calculate_walking_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> Optional[float]:
        """Calculate walking distance between two points using OSM"""
        cache_key = f"{point1[0]:.6f},{point1[1]:.6f}-{point2[0]:.6f},{point2[1]:.6f}"

        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Use geodesic distance as fallback if points are very close
        geodesic_dist = geodesic(point1, point2).meters
        if geodesic_dist < 50:  # Less than 50 meters
            self.distance_cache[cache_key] = geodesic_dist
            return geodesic_dist

        try:
            # Get center point for graph
            center_lat = (point1[0] + point2[0]) / 2
            center_lng = (point1[1] + point2[1]) / 2

            G = self._get_or_create_graph(center_lat, center_lng)
            if G is None:
                # Fallback to geodesic distance
                self.distance_cache[cache_key] = geodesic_dist
                return geodesic_dist

            # Find nearest nodes
            node1 = ox.distance.nearest_nodes(G, point1[1], point1[0])
            node2 = ox.distance.nearest_nodes(G, point2[1], point2[0])

            # Calculate shortest path
            try:
                path_length = nx.shortest_path_length(
                    G, node1, node2, weight='length')
                self.distance_cache[cache_key] = path_length

                # Limit cache size
                if len(self.distance_cache) > self.cache_size:
                    oldest_key = next(iter(self.distance_cache))
                    del self.distance_cache[oldest_key]

                return path_length
            except nx.NetworkXNoPath:
                # No walking path found, use geodesic distance
                self.distance_cache[cache_key] = geodesic_dist
                return geodesic_dist

        except Exception as e:
            logger.warning(f"Error calculating walking distance: {e}")
            # Fallback to geodesic distance
            self.distance_cache[cache_key] = geodesic_dist
            return geodesic_dist


class RideShareClusterer:
    """Main clustering algorithm for ride sharing"""

    def __init__(self, max_walking_distance: float = 500.0):  # 500 meters max walking
        self.max_walking_distance = max_walking_distance
        self.distance_calculator = OSMDistanceCalculator()

    def _calculate_compatibility_score(self, user1: UserLocation, user2: UserLocation) -> float:
        """Calculate compatibility score between two users"""
        try:
            # Calculate walking distances
            origin_distance = self.distance_calculator.calculate_walking_distance(
                user1.origin_coords, user2.origin_coords
            )
            destination_distance = self.distance_calculator.calculate_walking_distance(
                user1.destination_coords, user2.destination_coords
            )

            if origin_distance is None or destination_distance is None:
                return float('inf')

            # Check if within walking distance threshold
            if origin_distance > self.max_walking_distance or destination_distance > self.max_walking_distance:
                return float('inf')

            # Calculate combined score (lower is better)
            # Weight destination more heavily as it's the final meeting point
            score = (origin_distance * 0.6) + (destination_distance * 0.4)

            # Add time penalty if requests are far apart in time
            time_diff = abs(
                (user1.stored_at - user2.stored_at).total_seconds())
            # Max 100 meter penalty for 5+ min difference
            time_penalty = min(time_diff / 300, 100)

            return score + time_penalty

        except Exception as e:
            logger.error(f"Error calculating compatibility: {e}")
            return float('inf')

    def _find_best_third_user(self, user1: UserLocation, user2: UserLocation,
                              candidates: List[UserLocation]) -> Optional[UserLocation]:
        """Find the best third user to complete a group"""
        best_user = None
        best_score = float('inf')

        for candidate in candidates:
            if candidate.user_id in [user1.user_id, user2.user_id]:
                continue

            # Calculate scores with both existing users
            score1 = self._calculate_compatibility_score(user1, candidate)
            score2 = self._calculate_compatibility_score(user2, candidate)

            if score1 == float('inf') or score2 == float('inf'):
                continue

            # Combined score (average)
            combined_score = (score1 + score2) / 2

            if combined_score < best_score:
                best_score = combined_score
                best_user = candidate

        return best_user

    def cluster_users(self, users: List[UserLocation]) -> List[UserGroup]:
        """Main clustering algorithm"""
        if len(users) < 3:
            return []

        groups = []
        used_users = set()

        # Sort users by timestamp to prioritize earlier requests
        sorted_users = sorted(users, key=lambda u: u.stored_at)

        for i, user1 in enumerate(sorted_users):
            if user1.user_id in used_users:
                continue

            best_pair = None
            best_pair_score = float('inf')

            # Find best compatible user
            for j, user2 in enumerate(sorted_users[i+1:], i+1):
                if user2.user_id in used_users:
                    continue

                score = self._calculate_compatibility_score(user1, user2)
                if score < best_pair_score:
                    best_pair_score = score
                    best_pair = user2

            if best_pair is None or best_pair_score == float('inf'):
                continue

            # Find third user
            remaining_users = [
                u for u in sorted_users if u.user_id not in used_users]
            third_user = self._find_best_third_user(
                user1, best_pair, remaining_users)

            if third_user is not None:
                # Create group
                group = UserGroup(
                    users=[user1, best_pair, third_user],
                    created_at=datetime.now(timezone.utc),
                    group_id=f"group_{user1.user_id}_{best_pair.user_id}_{third_user.user_id}_{int(datetime.now().timestamp())}"
                )
                groups.append(group)

                # Mark users as used
                used_users.update(
                    [user1.user_id, best_pair.user_id, third_user.user_id])

        return groups


class RealTimeClusteringService:
    """Main service that manages real-time clustering"""

    def __init__(self, clustering_interval: int = 30, max_wait_time: int = 300):
        self.clustering_interval = clustering_interval  # seconds
        self.max_wait_time = max_wait_time  # 5 minutes max wait
        self.clusterer = RideShareClusterer()
        self.active_groups: Dict[str, UserGroup] = {}
        self.user_groups: Dict[int, str] = {}  # user_id -> group_id
        self.is_running = False
        self._observers: List[callable] = []

    def add_observer(self, callback: callable):
        """Add observer for group updates"""
        self._observers.append(callback)

    def remove_observer(self, callback: callable):
        """Remove observer"""
        if callback in self._observers:
            self._observers.remove(callback)

    def _notify_observers(self, event_type: str, data: dict):
        """Notify all observers of changes"""
        for callback in self._observers:
            try:
                asyncio.create_task(callback(event_type, data))
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

    def get_recent_locations(self, db: Session, max_age_minutes: int = 10) -> List[UserLocation]:
        """Get recent location requests from database"""
        cutoff_time = datetime.now(timezone.utc) - \
            timedelta(minutes=max_age_minutes)

        # Filter all locations newer than cutoff_time
        locations = db.query(Location).filter(
            Location.stored_at >= cutoff_time).all()

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

    def get_user_companions(self, user_id: int) -> Optional[Dict]:
        """Get companions for a specific user"""
        if user_id not in self.user_groups:
            return None

        group_id = self.user_groups[user_id]
        group = self.active_groups.get(group_id)

        if not group or not group.is_complete():
            return None

        companions = [u for u in group.users if u.user_id != user_id]
        return {
            'group_id': group_id,
            'companions': [
                {
                    'user_id': comp.user_id,
                    'origin_lat': comp.origin_lat,
                    'origin_lng': comp.origin_lng,
                    'destination_lat': comp.destination_lat,
                    'destination_lng': comp.destination_lng
                }
                for comp in companions
            ],
            'created_at': group.created_at.isoformat()
        }

    async def _clustering_loop(self):
        """Main clustering loop"""
        while self.is_running:
            try:
                db = SessionLocal()
                try:
                    # Get recent locations
                    recent_locations = self.get_recent_locations(db)

                    # Filter out users who are already in complete groups
                    available_users = [
                        user for user in recent_locations
                        if user.user_id not in self.user_groups or
                        not self.active_groups.get(self.user_groups[user.user_id], UserGroup(
                            [], datetime.now(), "")).is_complete()
                    ]

                    if len(available_users) >= 3:
                        # Perform clustering
                        new_groups = self.clusterer.cluster_users(
                            available_users)

                        # Process new groups
                        for group in new_groups:
                            self.active_groups[group.group_id] = group

                            # Update user mappings
                            for user in group.users:
                                self.user_groups[user.user_id] = group.group_id

                            # Notify observers
                            self._notify_observers('group_formed', {
                                'group_id': group.group_id,
                                'users': [u.user_id for u in group.users]
                            })

                            logger.info(
                                f"Formed new group {group.group_id} with users: {[u.user_id for u in group.users]}")

                    # Clean up old incomplete groups
                    current_time = datetime.now(timezone.utc)
                    expired_groups = []

                    for group_id, group in self.active_groups.items():
                        if not group.is_complete() and (current_time - group.created_at).total_seconds() > self.max_wait_time:
                            expired_groups.append(group_id)

                    for group_id in expired_groups:
                        group = self.active_groups[group_id]
                        for user in group.users:
                            if user.user_id in self.user_groups:
                                del self.user_groups[user.user_id]
                        del self.active_groups[group_id]

                        self._notify_observers(
                            'group_expired', {'group_id': group_id})
                        logger.info(f"Expired incomplete group {group_id}")

                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Error in clustering loop: {e}")

            await asyncio.sleep(self.clustering_interval)

    async def start(self):
        """Start the clustering service"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting real-time clustering service")
        await self._clustering_loop()

    def stop(self):
        """Stop the clustering service"""
        self.is_running = False
        logger.info("Stopping real-time clustering service")

    def get_service_status(self) -> Dict:
        """Get current service status"""
        return {
            'is_running': self.is_running,
            'active_groups': len(self.active_groups),
            'complete_groups': len([g for g in self.active_groups.values() if g.is_complete()]),
            'users_in_groups': len(self.user_groups)
        }


# Global service instance
clustering_service = RealTimeClusteringService()


async def start_clustering_service():
    """Start the clustering service (call this in FastAPI startup)"""
    asyncio.create_task(clustering_service.start())


def get_clustering_service() -> RealTimeClusteringService:
    """Get the clustering service instance"""
    return clustering_service
