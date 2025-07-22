# main.py
from fastapi import FastAPI, Depends, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import os

# Initialize FastAPI app
app = FastAPI(title="RideShare App Backend")

# Secret key for JWT. In a real application, use a strong, randomly generated key
# and store it securely (e.g., in environment variables).
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-replace-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# In-memory "database" for users. In a real application, use a proper database (e.g., PostgreSQL, MongoDB).
# This dictionary stores user data: {"username": {"username": "...", "hashed_password": "..."}}
users_db = {}

# --- Utility Functions for Security ---


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    """Decodes a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# --- User Management Functions ---


def get_user(username: str):
    """Retrieves a user from the in-memory database."""
    return users_db.get(username)


def authenticate_user(username: str, password: str):
    """Authenticates a user by checking username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

# --- Routes ---


@app.post("/register", tags=["Authentication"])
async def register_user(username: str = Form(...), password: str = Form(...)):
    """
    Registers a new user.
    Expects form data: username, password.
    """
    if get_user(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    hashed_password = get_password_hash(password)
    users_db[username] = {"username": username,
                          "hashed_password": hashed_password}
    return {"message": "User registered successfully"}


@app.post("/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Handles user login and issues a JWT access token.
    Expects form data: username, password.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", tags=["Users"])
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Retrieves information about the current authenticated user.
    Requires a valid JWT in the Authorization header.
    """
    return {"username": current_user["username"]}


@app.get("/api/data/ride", tags=["API Data"])
async def get_ride_data(current_user: dict = Depends(get_current_user)):
    """
    Example protected endpoint for ride data.
    """
    return {"message": f"Welcome, {current_user['username']}! Here's your ride data."}


@app.get("/api/data/schedule", tags=["API Data"])
async def get_schedule_data(current_user: dict = Depends(get_current_user)):
    """
    Example protected endpoint for schedule data.
    """
    return {"message": f"Welcome, {current_user['username']}! Here's your schedule data."}


@app.get("/api/data/history", tags=["API Data"])
async def get_history_data(current_user: dict = Depends(get_current_user)):
    """
    Example protected endpoint for history data.
    """
    return {"message": f"Welcome, {current_user['username']}! Here's your history data."}


# --- Serve Frontend Files ---
# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for serving HTML
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_login_page(request: Request):
    """Serves the login page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/home", response_class=HTMLResponse, include_in_schema=False)
async def serve_home_page(request: Request):
    """Serves the homepage (requires client-side token check)."""
    return templates.TemplateResponse("home.html", {"request": request})

# To run this FastAPI app:
# 1. Save the code above as `main.py`.
# 2. Create a folder named `templates` in the same directory as `main.py`.
# 3. Create a folder named `static` inside the main directory.
# 4. Inside `templates`, create `index.html` (for login) and `home.html` (for homepage).
# 5. Inside `static`, you can place CSS or JS files if needed (though for this example, most JS is inline in HTML).
# 6. Install dependencies: `pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] python-multipart jinja2`
# 7. Run the app: `uvicorn main:app --reload`
