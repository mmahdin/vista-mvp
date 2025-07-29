from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.templates_utils import templates
from app.database.crud import create_location, get_all_locations_as_dataframe
from app.database.base import get_db
from app.database.models import *
from typing import Annotated, List
from datetime import datetime, timezone
from .utils import get_od_meeting_points
from .utils import add_random_data

router = APIRouter()


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    phone_number: str | None
    is_active: bool
    created_at: datetime | None

    class Config:
        from_attributes = True


@router.get("/ride", response_class=HTMLResponse)
async def ride_page(
    request: Request,
    user=Depends(protected_route)
):
    # await add_random_data()

    user_response = UserResponse.model_validate(user)
    return templates.TemplateResponse("ride.html", {
        "request": request,
        "user": user_response
    })


@router.post("/request-ride")
async def request_ride(
    request: Request,
    user=Depends(protected_route)
):
    form_data = await request.form()

    # For demo, just return success
    return {"message": "Ride requested successfully",
            "start": form_data.get("start_location"),
            "end": form_data.get("end_location")}


# =====================================================================================


class LocationHistoryCreate(BaseModel):
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float


class LocationHistoryResponse(BaseModel):
    id: int
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    stored_at: datetime

    class Config:
        from_attributes = True


@router.post("/save-location/", response_model=List[LocationHistoryResponse])
async def save_location_history(
    location_data: LocationHistoryCreate,
    db: Annotated[Session, Depends(get_db)]
):
    try:
        # Check if user exists
        user = db.query(User).filter(User.id == location_data.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Create new location history
        create_location(
            db=db,
            user_id=location_data.user_id,
            origin_lat=location_data.origin_lat,
            origin_lng=location_data.origin_lng,
            destination_lat=location_data.destination_lat,
            destination_lng=location_data.destination_lng
        )

        # Get locations and calculate groups
        df_locations = get_all_locations_as_dataframe(db)
        meeting_points, groups = get_od_meeting_points(
            df_locations,
            group_size=3,
            origin_weight=0.6,
            dest_weight=0.4,
            max_distance=800
        )

        # Find groups containing the current user
        matching_groups = [lst for lst in groups if user.id in lst][0]
        matching_groups.remove(user.id)

        ride_users = []

        # Process each matching group
        for idx in matching_groups:
            location = df_locations[df_locations['user_id'] == idx].squeeze()
            ride_users.append({
                "id": location.id,
                "user_id": location.user_id,
                "origin_lat": location.origin_lat,
                "origin_lng": location.origin_lng,
                "destination_lat": location.destination_lat,
                "destination_lng": location.destination_lng,
                "stored_at": location.stored_at
            })
        return ride_users

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving location: {str(e)}"
        )
# =====================================================================================


@router.get("/api/random-locations/")
async def get_random_locations(db: Session = Depends(get_db)):
    """
    Get all random location data from the database
    """
    try:
        locations = db.query(Location).all()

        location_data = []
        for location in locations:
            if location.user_id > 10000:
                location_data.append({
                    "id": location.id,
                    "user_id": location.user_id,
                    "origin_lat": location.origin_lat,
                    "origin_lng": location.origin_lng,
                    "destination_lat": location.destination_lat,
                    "destination_lng": location.destination_lng,
                    "stored_at": location.stored_at.isoformat() if location.stored_at else None
                })

        return {
            "success": True,
            "data": location_data,
            "count": len(location_data)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")
