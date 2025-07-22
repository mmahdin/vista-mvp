from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import uvicorn

app = FastAPI(title="RideShare App", version="1.0.0")

# Security configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-memory user storage (use database in production)
fake_users_db = {
    "demo@example.com": {
        "id": 1,
        "email": "demo@example.com",
        "hashed_password": pwd_context.hash("demo123"),
        "name": "Demo User",
        "phone": "+1234567890"
    }
}

# Pydantic models


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str
    phone: str


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    id: int
    email: str
    name: str
    phone: str


class RideRequest(BaseModel):
    pickup_location: str
    destination: str
    ride_type: str  # "economy", "premium", "shared"
    passenger_count: int = 1


class ScheduleRide(BaseModel):
    pickup_location: str
    destination: str
    ride_type: str
    passenger_count: int
    scheduled_time: datetime

# Authentication functions


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def authenticate_user(email: str, password: str):
    user = fake_users_db.get(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.token, SECRET_KEY,
                             algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = fake_users_db.get(email)
    if user is None:
        raise credentials_exception
    return user

# Routes


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RideShare - Modern Transportation</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                padding: 40px;
                width: 100%;
                max-width: 400px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .homepage {
                max-width: 900px;
                width: 100%;
                padding: 40px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 24px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            }
            
            .logo {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .logo h1 {
                color: #667eea;
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 8px;
            }
            
            .logo p {
                color: #6b7280;
                font-size: 1.1em;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: #374151;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 15px;
                border: 2px solid #e5e7eb;
                border-radius: 12px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: rgba(255, 255, 255, 0.9);
            }
            
            input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                transform: translateY(-1px);
            }
            
            .btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 15px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .toggle-form {
                text-align: center;
                margin-top: 20px;
            }
            
            .toggle-form a {
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }
            
            .toggle-form a:hover {
                text-decoration: underline;
            }
            
            .hidden {
                display: none;
            }
            
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e5e7eb;
            }
            
            .user-info {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .user-avatar {
                width: 50px;
                height: 50px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
                font-size: 18px;
            }
            
            .logout-btn {
                background: #ef4444;
                padding: 8px 16px;
                border: none;
                border-radius: 8px;
                color: white;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .logout-btn:hover {
                background: #dc2626;
                transform: translateY(-1px);
            }
            
            .ride-options {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 30px;
                margin-bottom: 30px;
            }
            
            .ride-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                padding: 30px;
                color: white;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .ride-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
            }
            
            .ride-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(255, 255, 255, 0.1);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .ride-card:hover::before {
                opacity: 1;
            }
            
            .ride-icon {
                font-size: 3em;
                margin-bottom: 15px;
                display: block;
            }
            
            .ride-title {
                font-size: 1.5em;
                font-weight: 600;
                margin-bottom: 10px;
            }
            
            .ride-description {
                opacity: 0.9;
                line-height: 1.5;
            }
            
            .quick-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.7);
                padding: 20px;
                border-radius: 16px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            .stat-number {
                font-size: 2em;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }
            
            .stat-label {
                color: #6b7280;
                font-weight: 500;
            }
            
            .alert {
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: 500;
            }
            
            .alert-error {
                background: #fef2f2;
                color: #dc2626;
                border: 1px solid #fecaca;
            }
            
            .alert-success {
                background: #f0fdf4;
                color: #16a34a;
                border: 1px solid #bbf7d0;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .container, .homepage {
                animation: slideIn 0.6s ease-out;
            }
        </style>
    </head>
    <body>
        <!-- Login Form -->
        <div id="loginContainer" class="container">
            <div class="logo">
                <h1>🚗 RideShare</h1>
                <p>Modern Transportation Solutions</p>
            </div>
            
            <div id="alertContainer"></div>
            
            <form id="loginForm">
                <div class="form-group">
                    <label for="email">Email Address</label>
                    <input type="email" id="email" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" required>
                </div>
                
                <button type="submit" class="btn">Sign In</button>
            </form>
            
            <div class="toggle-form">
                <p>Don't have an account? <a href="#" id="showRegister">Sign up here</a></p>
                <p style="margin-top: 10px; font-size: 14px; color: #6b7280;">
                    Demo: demo@example.com / demo123
                </p>
            </div>
        </div>
        
        <!-- Register Form -->
        <div id="registerContainer" class="container hidden">
            <div class="logo">
                <h1>🚗 RideShare</h1>
                <p>Join Our Community</p>
            </div>
            
            <div id="registerAlertContainer"></div>
            
            <form id="registerForm">
                <div class="form-group">
                    <label for="regName">Full Name</label>
                    <input type="text" id="regName" required>
                </div>
                
                <div class="form-group">
                    <label for="regEmail">Email Address</label>
                    <input type="email" id="regEmail" required>
                </div>
                
                <div class="form-group">
                    <label for="regPhone">Phone Number</label>
                    <input type="tel" id="regPhone" required>
                </div>
                
                <div class="form-group">
                    <label for="regPassword">Password</label>
                    <input type="password" id="regPassword" required>
                </div>
                
                <button type="submit" class="btn">Create Account</button>
            </form>
            
            <div class="toggle-form">
                <p>Already have an account? <a href="#" id="showLogin">Sign in here</a></p>
            </div>
        </div>
        
        <!-- Homepage -->
        <div id="homepage" class="homepage hidden">
            <div class="header">
                <div>
                    <h1 style="color: #667eea; font-size: 2em;">🚗 RideShare</h1>
                    <p style="color: #6b7280;">Welcome back!</p>
                </div>
                <div class="user-info">
                    <div class="user-avatar" id="userAvatar">U</div>
                    <div>
                        <div style="font-weight: 600; color: #374151;" id="userName">User</div>
                        <div style="font-size: 14px; color: #6b7280;">Premium Member</div>
                    </div>
                    <button class="logout-btn" onclick="logout()">Logout</button>
                </div>
            </div>
            
            <div class="ride-options">
                <div class="ride-card" onclick="showRideModal('instant')">
                    <div class="ride-icon">🚗</div>
                    <div class="ride-title">Ride Now</div>
                    <div class="ride-description">Get a ride instantly to your destination</div>
                </div>
                
                <div class="ride-card" onclick="showRideModal('schedule')">
                    <div class="ride-icon">📅</div>
                    <div class="ride-title">Schedule Ride</div>
                    <div class="ride-description">Plan your trip ahead of time</div>
                </div>
                
                <div class="ride-card" onclick="showHistory()">
                    <div class="ride-icon">📊</div>
                    <div class="ride-title">Ride History</div>
                    <div class="ride-description">View your past trips and receipts</div>
                </div>
            </div>
            
            <div class="quick-stats">
                <div class="stat-card">
                    <div class="stat-number">24</div>
                    <div class="stat-label">Total Rides</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">$156</div>
                    <div class="stat-label">Total Saved</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">4.9★</div>
                    <div class="stat-label">Your Rating</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">2.5k</div>
                    <div class="stat-label">Miles Traveled</div>
                </div>
            </div>
        </div>
        
        <script>
            let currentUser = null;
            
            // Check if user is already logged in
            window.onload = function() {
                const token = localStorage.getItem('token');
                if (token) {
                    verifyToken(token);
                }
            };
            
            // Form toggle functions
            document.getElementById('showRegister').onclick = function(e) {
                e.preventDefault();
                document.getElementById('loginContainer').classList.add('hidden');
                document.getElementById('registerContainer').classList.remove('hidden');
            };
            
            document.getElementById('showLogin').onclick = function(e) {
                e.preventDefault();
                document.getElementById('registerContainer').classList.add('hidden');
                document.getElementById('loginContainer').classList.remove('hidden');
            };
            
            // Login form submission
            document.getElementById('loginForm').onsubmit = async function(e) {
                e.preventDefault();
                const email = document.getElementById('email').value;
                const password = document.getElementById('password').value;
                
                try {
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email, password })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        localStorage.setItem('token', data.access_token);
                        showHomepage(data.user);
                    } else {
                        const error = await response.json();
                        showAlert('alertContainer', error.detail, 'error');
                    }
                } catch (error) {
                    showAlert('alertContainer', 'Network error. Please try again.', 'error');
                }
            };
            
            // Register form submission
            document.getElementById('registerForm').onsubmit = async function(e) {
                e.preventDefault();
                const name = document.getElementById('regName').value;
                const email = document.getElementById('regEmail').value;
                const phone = document.getElementById('regPhone').value;
                const password = document.getElementById('regPassword').value;
                
                try {
                    const response = await fetch('/auth/register', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name, email, phone, password })
                    });
                    
                    if (response.ok) {
                        showAlert('registerAlertContainer', 'Account created successfully! Please sign in.', 'success');
                        setTimeout(() => {
                            document.getElementById('registerContainer').classList.add('hidden');
                            document.getElementById('loginContainer').classList.remove('hidden');
                        }, 2000);
                    } else {
                        const error = await response.json();
                        showAlert('registerAlertContainer', error.detail, 'error');
                    }
                } catch (error) {
                    showAlert('registerAlertContainer', 'Network error. Please try again.', 'error');
                }
            };
            
            function showAlert(containerId, message, type) {
                const container = document.getElementById(containerId);
                container.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
                setTimeout(() => container.innerHTML = '', 5000);
            }
            
            function showHomepage(user) {
                currentUser = user;
                document.getElementById('loginContainer').classList.add('hidden');
                document.getElementById('registerContainer').classList.add('hidden');
                document.getElementById('homepage').classList.remove('hidden');
                
                // Update user info
                document.getElementById('userName').textContent = user.name;
                document.getElementById('userAvatar').textContent = user.name.charAt(0).toUpperCase();
            }
            
            async function verifyToken(token) {
                try {
                    const response = await fetch('/auth/me', {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    
                    if (response.ok) {
                        const user = await response.json();
                        showHomepage(user);
                    } else {
                        localStorage.removeItem('token');
                    }
                } catch (error) {
                    localStorage.removeItem('token');
                }
            }
            
            function logout() {
                localStorage.removeItem('token');
                currentUser = null;
                document.getElementById('homepage').classList.add('hidden');
                document.getElementById('loginContainer').classList.remove('hidden');
            }
            
            function showRideModal(type) {
                if (type === 'instant') {
                    alert('🚗 Instant Ride feature coming soon!\\n\\nWe\\'re working on connecting you with nearby drivers.');
                } else if (type === 'schedule') {
                    alert('📅 Schedule Ride feature coming soon!\\n\\nYou\\'ll be able to book rides in advance.');
                }
            }
            
            function showHistory() {
                alert('📊 Ride History feature coming soon!\\n\\nView all your past trips, receipts, and statistics.');
            }
        </script>
    </body>
    </html>
    """


@app.post("/auth/login", response_model=Token)
async def login_for_access_token(user_credentials: UserLogin):
    user = authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "phone": user["phone"]
        }
    }


@app.post("/auth/register")
async def register_user(user_data: UserRegister):
    if user_data.email in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    new_user_id = max([user["id"]
                      for user in fake_users_db.values()], default=0) + 1
    hashed_password = get_password_hash(user_data.password)

    fake_users_db[user_data.email] = {
        "id": new_user_id,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "name": user_data.name,
        "phone": user_data.phone
    }

    return {"message": "User registered successfully"}


@app.get("/auth/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return User(
        id=current_user["id"],
        email=current_user["email"],
        name=current_user["name"],
        phone=current_user["phone"]
    )


@app.post("/rides/request")
async def request_ride(ride_request: RideRequest, current_user: dict = Depends(get_current_user)):
    # This would integrate with your ride-matching logic
    return {
        "message": "Ride requested successfully",
        "ride_id": "ride_123",
        "estimated_arrival": "5-8 minutes",
        "fare_estimate": "$12-15",
        "driver_info": {
            "name": "John Doe",
            "rating": 4.8,
            "vehicle": "Toyota Camry - ABC 123"
        }
    }


@app.post("/rides/schedule")
async def schedule_ride(ride_schedule: ScheduleRide, current_user: dict = Depends(get_current_user)):
    return {
        "message": "Ride scheduled successfully",
        "schedule_id": "sched_456",
        "scheduled_time": ride_schedule.scheduled_time,
        "fare_estimate": "$12-15"
    }


@app.get("/rides/history")
async def get_ride_history(current_user: dict = Depends(get_current_user)):
    # Mock ride history data
    return {
        "rides": [
            {
                "id": "ride_001",
                "date": "2024-01-20",
                "from": "Downtown Plaza",
                "to": "Airport Terminal 1",
                "fare": 25.50,
                "status": "completed",
                "driver": "Sarah Johnson",
                "rating": 5
            },
            {
                "id": "ride_002",
                "date": "2024-01-18",
                "from": "Home",
                "to": "Shopping Mall",
                "fare": 12.75,
                "status": "completed",
                "driver": "Mike Wilson",
                "rating": 4
            }
        ],
        "total_rides": 24,
        "total_spent": 456.75,
        "average_rating": 4.9
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
