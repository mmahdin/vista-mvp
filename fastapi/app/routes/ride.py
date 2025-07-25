from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.dependencies import protected_route
from app.templates_utils import templates

router = APIRouter()


@router.get("/ride", response_class=HTMLResponse)
async def ride_page(
    request: Request,
    user=Depends(protected_route)
):
    print(user)
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
