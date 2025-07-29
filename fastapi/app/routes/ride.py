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
from .utils import _add_random_data, get_od_meeting_points

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
    _add_random_data()

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
    # In a real app, you would:
    # 1. Validate form data
    # 2. Create ride in database
    # 3. Process payment
    # 4. Notify drivers

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


@router.post("/save-location/", response_model=LocationHistoryResponse)
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

        location_history = create_location(
            db=db,
            user_id=location_data.user_id,
            origin_lat=location_data.origin_lat,
            origin_lng=location_data.origin_lng,
            destination_lat=location_data.destination_lat,
            destination_lng=location_data.destination_lng
        )

        df_locations = get_all_locations_as_dataframe(db)
        meeting_points, groups = get_od_meeting_points(
            df_locations,
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

        return location_history

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
