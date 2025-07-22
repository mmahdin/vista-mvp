from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from app.database.base import get_db
from app.database.crud import get_user_by_email, create_user
from .utils import create_access_token
from app.security import verify_password
from app.templates_utils import templates

router = APIRouter(tags=["Authentication"])


@router.post("/token")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login")
async def login(
    request: Request,
    db: Session = Depends(get_db)
):
    form_data = await request.form()
    user = get_user_by_email(db, form_data["email"])

    if not user or not verify_password(form_data["password"], user.hashed_password):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid email or password"
        })

    access_token = create_access_token(data={"sub": user.email})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@router.post("/signup")
async def signup(
    request: Request,
    db: Session = Depends(get_db)
):
    form_data = await request.form()

    # Check if user already exists
    if get_user_by_email(db, form_data["email"]):
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Email already registered"
        })

    # Create new user
    create_user(
        db,
        email=form_data["email"],
        password=form_data["password"],
        full_name=form_data["full_name"]
    )

    # Auto-login after signup
    access_token = create_access_token(data={"sub": form_data["email"]})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response
