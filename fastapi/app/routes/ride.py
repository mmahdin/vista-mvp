from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.templates_utils import templates
from app.database.crud import create_location, get_all_locations_as_dataframe
from app.database.base import get_db
from app.database.models import *
from typing import Annotated
from datetime import datetime, timezone
# from utils import *

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


# ==================================================================


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

        return location_history

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving location: {str(e)}"
        )
