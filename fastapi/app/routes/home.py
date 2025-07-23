from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse

from app.dependencies import get_current_user
from app.templates_utils import templates
from app.database.models import User

from typing import Annotated

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home_page(
    request: Request,
    user: Annotated[User, Depends(get_current_user)]
):
    features = [
        {
            "name": "Ride",
            "icon": "🚗",
            "url": "/ride",
            "description": "Request a ride now"
        },
        {
            "name": "Schedule",
            "icon": "📅",
            "url": "/schedule",
            "description": "Plan your future trips"
        },
        {
            "name": "History",
            "icon": "📝",
            "url": "/history",
            "description": "View your ride history"
        }
    ]

    return templates.TemplateResponse("home.html", {
        "request": request,
        "user": user,
        "features": features
    })
