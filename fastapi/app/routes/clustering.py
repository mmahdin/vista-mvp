# routes/clustering.py
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

from app.database import get_db
from app.database.models import *
from .clustering_service import get_clustering_service, RealTimeClusteringService

router = APIRouter(prefix="/clustering", tags=["clustering"])
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}  # user_id -> websocket
    
    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected to clustering WebSocket")
    
    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected from clustering WebSocket")
    
    async def send_personal_message(self, message: dict, user_id: int):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)
    
    async def send_group_message(self, message: dict, user_ids: List[int]):
        for user_id in user_ids:
            await self.send_personal_message(message, user_id)

manager = ConnectionManager()

# Observer function for clustering service
async def clustering_observer(event_type: str, data: dict):
    """Handle clustering events and notify connected users"""
    if event_type == "group_formed":
        group_id = data['group_id']
        user_ids = data['users']
        
        # Get group details
        clustering_service = get_clustering_service()
        
        for user_id in user_ids:
            companions_data = clustering_service.get_user_companions(user_id)
            if companions_data:
                message = {
                    'type': 'companions_found',
                    'data': companions_data
                }
                await manager.send_personal_message(message, user_id)
    
    elif event_type == "group_expired":
        # Notify users that their group search expired
        # This could be enhanced to include user_ids if needed
        pass

# Add observer to clustering service
get_clustering_service().add_observer(clustering_observer)

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time clustering updates"""
    await manager.connect(websocket, user_id)
    try:
        # Send current companions if they exist
        clustering_service = get_clustering_service()
        companions_data = clustering_service.get_user_companions(user_id)
        
        if companions_data:
            initial_message = {
                'type': 'companions_found',
                'data': companions_data
            }
            await websocket.send_text(json.dumps(initial_message))
        else:
            # Send waiting message
            waiting_message = {
                'type': 'waiting_for_companions',
                'data': {'message': 'Searching for companions...'}
            }
            await websocket.send_text(json.dumps(waiting_message))
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client (heartbeat, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                elif message.get('type') == 'request_update':
                    # Send current companions status
                    companions_data = clustering_service.get_user_companions(user_id)
                    if companions_data:
                        response = {
                            'type': 'companions_found',
                            'data': companions_data
                        }
                    else:
                        response = {
                            'type': 'waiting_for_companions',
                            'data': {'message': 'Still searching for companions...'}
                        }
                    await websocket.send_text(json.dumps(response))
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket communication with user {user_id}: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        manager.disconnect(user_id)

@router.get("/companions/{user_id}")
async def get_user_companions(user_id: int):
    """Get companions for a specific user (HTTP endpoint)"""
    clustering_service = get_clustering_service()
    companions_data = clustering_service.get_user_companions(user_id)
    
    if companions_data:
        return JSONResponse(content={
            'status': 'found',
            'data': companions_data
        })
    else:
        return JSONResponse(content={
            'status': 'searching',
            'message': 'Still searching for companions'
        })

@router.get("/status")
async def get_clustering_status():
    """Get clustering service status"""
    clustering_service = get_clustering_service()
    status = clustering_service.get_service_status()
    return JSONResponse(content=status)

@router.delete("/group/{user_id}")
async def leave_group(user_id: int):
    """Allow user to leave their current group"""
    clustering_service = get_clustering_service()
    
    if user_id in clustering_service.user_groups:
        group_id = clustering_service.user_groups[user_id]
        group = clustering_service.active_groups.get(group_id)
        
        if group:
            # Remove user from group
            group.users = [u for u in group.users if u.user_id != user_id]
            del clustering_service.user_groups[user_id]
            
            # If group becomes empty, remove it
            if len(group.users) == 0:
                del clustering_service.active_groups[group_id]
            
            # Notify remaining users
            remaining_user_ids = [u.user_id for u in group.users]
            if remaining_user_ids:
                message = {
                    'type': 'user_left_group',
                    'data': {
                        'user_id': user_id,
                        'remaining_users': len(group.users)
                    }
                }
                await manager.send_group_message(message, remaining_user_ids)
            
            return JSONResponse(content={'status': 'success', 'message': 'Left group successfully'})
    
    return JSONResponse(content={'status': 'error', 'message': 'User not in any group'})

# main.py additions
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    from app.routes.clustering_service import start_clustering_service
    await start_clustering_service()
    logger.info("Clustering service started")
    
    yield
    
    # Shutdown
    from app.routes.clustering_service import get_clustering_service
    get_clustering_service().stop()
    logger.info("Clustering service stopped")

# When creating your FastAPI app:
# app = FastAPI(lifespan=lifespan)
# app.include_router(clustering_router)

# Updated save-location endpoint
@router.post("/save-location/")
async def save_location_enhanced(location_data: dict, db: Session = Depends(get_db)):
    """Enhanced save location endpoint that integrates with clustering"""
    try:
        # Save location to database (your existing logic)
        new_location = Location(
            user_id=location_data["user_id"],
            origin_lat=location_data["origin_lat"],
            origin_lng=location_data["origin_lng"],
            destination_lat=location_data["destination_lat"],
            destination_lng=location_data["destination_lng"]
        )
        
        db.add(new_location)
        db.commit()
        db.refresh(new_location)
        
        # Trigger immediate clustering check for this user
        clustering_service = get_clustering_service()
        
        # Send immediate update to user's WebSocket if connected
        if location_data["user_id"] in manager.active_connections:
            message = {
                'type': 'location_saved',
                'data': {
                    'message': 'Location saved, searching for companions...',
                    'timestamp': datetime.now().isoformat()
                }
            }
            await manager.send_personal_message(message, location_data["user_id"])
        
        return JSONResponse(content={
            'status': 'success',
            'message': 'Location saved successfully',
            'location_id': new_location.id
        })
        
    except Exception as e:
        logger.error(f"Error saving location: {e}")
        raise HTTPException(status_code=500, detail="Failed to save location")