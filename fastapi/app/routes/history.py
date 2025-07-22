from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.database.base import get_db  # Import directly from base
from sqlalchemy.orm import Session
from app.database.crud import get_user_rides  # Import CRUD directly
from app.templates_utils import templates
router = APIRouter()


@router.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    user=Depends(protected_route),
    db: Session = Depends(get_db)
):
    rides = get_user_rides(db, user.id) if user else []
    return templates.TemplateResponse("history.html", {
        "request": request,
        "user": user,
        "rides": rides
    })
