from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.templates_utils import templates

router = APIRouter()


@router.get("/schedule", response_class=HTMLResponse)
async def schedule_page(
    request: Request,
    user=Depends(protected_route)
):
    return templates.TemplateResponse("schedule.html", {
        "request": request,
        "user": user
    })
