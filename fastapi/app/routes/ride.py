from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.templates_utils import templates
from app.database.crud import create_location_history
from app.database.base import get_db
from typing import Annotated
from datetime import datetime, timezone

router = APIRouter()


@router.get("/ride", response_class=HTMLResponse)
async def ride_page(
    request: Request,
    user=Depends(protected_route)
):
    return templates.TemplateResponse("ride.html", {
        "request": request,
        "user": user
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
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float


class LocationHistoryResponse(BaseModel):
    id: int
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
        location_history = create_location_history(
            db=db,
            origin_lat=location_data.origin_lat,
            origin_lng=location_data.origin_lng,
            destination_lat=location_data.destination_lat,
            destination_lng=location_data.destination_lng
        )
        return location_history
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error saving location history: {str(e)}")
