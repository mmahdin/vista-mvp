from fastapi import FastAPI, HTTPException, Depends, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import sqlite3
import hashlib
import os
from typing import Optional, List
import uvicorn

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(title="RideShare Pro",
              description="Modern Ride-Sharing Platform")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Pydantic Models


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str


class RideRequest(BaseModel):
    pickup_location: str
    destination: str
    ride_type: str
    scheduled_time: Optional[datetime] = None


class RideHistory(BaseModel):
    id: int
    pickup_location: str
    destination: str
    ride_type: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

# Database setup


def init_db():
    conn = sqlite3.connect('rideshare.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Rides table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            pickup_location TEXT NOT NULL,
            destination TEXT NOT NULL,
            ride_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            scheduled_time TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

# Database helper functions


def get_db_connection():
    conn = sqlite3.connect('rideshare.db')
    conn.row_factory = sqlite3.Row
    return conn


def get_user_by_username(username: str):
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return user


def create_user_in_db(user_data: UserCreate):
    conn = get_db_connection()
    hashed_password = pwd_context.hash(user_data.password)

    try:
        cursor = conn.execute(
            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
            (user_data.username, user_data.email,
             hashed_password, user_data.full_name)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(
            status_code=400, detail="Username or email already exists")

    conn.close()
    return user_id

# Authentication functions


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

# API Routes


@app.post("/api/register", response_model=dict)
async def register(user: UserCreate):
    user_id = create_user_in_db(user)
    return {"message": "User created successfully", "user_id": user_id}


@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    db_user = get_user_by_username(user.username)
    if not db_user or not verify_password(user.password, db_user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/user/me", response_model=User)
async def read_users_me(current_user=Depends(get_current_user)):
    return User(
        id=current_user['id'],
        username=current_user['username'],
        email=current_user['email'],
        full_name=current_user['full_name']
    )


@app.post("/api/rides/request")
async def request_ride(ride: RideRequest, current_user=Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.execute(
        """INSERT INTO rides (user_id, pickup_location, destination, ride_type, scheduled_time)
           VALUES (?, ?, ?, ?, ?)""",
        (current_user['id'], ride.pickup_location, ride.destination,
         ride.ride_type, ride.scheduled_time)
    )
    conn.commit()
    ride_id = cursor.lastrowid
    conn.close()

    return {"message": "Ride requested successfully", "ride_id": ride_id}


@app.get("/api/rides/history", response_model=List[RideHistory])
async def get_ride_history(current_user=Depends(get_current_user)):
    conn = get_db_connection()
    rides = conn.execute(
        "SELECT * FROM rides WHERE user_id = ? ORDER BY created_at DESC",
        (current_user['id'],)
    ).fetchall()
    conn.close()

    return [
        RideHistory(
            id=ride['id'],
            pickup_location=ride['pickup_location'],
            destination=ride['destination'],
            ride_type=ride['ride_type'],
            status=ride['status'],
            created_at=ride['created_at'],
            completed_at=ride['completed_at']
        ) for ride in rides
    ]

# Serve static files


@app.get("/", response_class=HTMLResponse)
async def serve_login():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RideShare Pro - Login</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .login-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                width: 100%;
                max-width: 400px;
                transition: transform 0.3s ease;
            }
            
            .login-container:hover {
                transform: translateY(-5px);
            }
            
            .logo {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .logo h1 {
                color: #333;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 5px;
            }
            
            .logo p {
                color: #666;
                font-size: 0.9rem;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
            }
            
            .form-group input {
                width: 100%;
                padding: 15px;
                border: 2px solid #e1e5e9;
                border-radius: 12px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: #f8f9fa;
            }
            
            .form-group input:focus {
                outline: none;
                border-color: #667eea;
                background: white;
                transform: scale(1.02);
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
            
            .switch-form {
                text-align: center;
                color: #666;
            }
            
            .switch-form a {
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }
            
            .switch-form a:hover {
                text-decoration: underline;
            }
            
            .error {
                color: #e74c3c;
                font-size: 14px;
                margin-top: 10px;
                text-align: center;
            }
            
            .success {
                color: #27ae60;
                font-size: 14px;
                margin-top: 10px;
                text-align: center;
            }
            
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="logo">
                <h1>🚗 RideShare</h1>
                <p>Your journey begins here</p>
            </div>
            
            <!-- Login Form -->
            <form id="loginForm">
                <div class="form-group">
                    <label for="username">Username</label>
                    <input type="text" id="username" name="username" required>
                </div>
                
                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>
                
                <button type="submit" class="btn">Sign In</button>
                
                <div class="switch-form">
                    Don't have an account? <a href="#" onclick="showRegister()">Sign Up</a>
                </div>
            </form>
            
            <!-- Register Form -->
            <form id="registerForm" class="hidden">
                <div class="form-group">
                    <label for="reg_username">Username</label>
                    <input type="text" id="reg_username" name="username" required>
                </div>
                
                <div class="form-group">
                    <label for="reg_email">Email</label>
                    <input type="email" id="reg_email" name="email" required>
                </div>
                
                <div class="form-group">
                    <label for="reg_fullname">Full Name</label>
                    <input type="text" id="reg_fullname" name="full_name" required>
                </div>
                
                <div class="form-group">
                    <label for="reg_password">Password</label>
                    <input type="password" id="reg_password" name="password" required>
                </div>
                
                <button type="submit" class="btn">Create Account</button>
                
                <div class="switch-form">
                    Already have an account? <a href="#" onclick="showLogin()">Sign In</a>
                </div>
            </form>
            
            <div id="message"></div>
        </div>
        
        <script>
            function showRegister() {
                document.getElementById('loginForm').classList.add('hidden');
                document.getElementById('registerForm').classList.remove('hidden');
                document.getElementById('message').innerHTML = '';
            }
            
            function showLogin() {
                document.getElementById('registerForm').classList.add('hidden');
                document.getElementById('loginForm').classList.remove('hidden');
                document.getElementById('message').innerHTML = '';
            }
            
            function showMessage(message, type = 'error') {
                const messageDiv = document.getElementById('message');
                messageDiv.innerHTML = `<div class="${type}">${message}</div>`;
            }
            
            // Login form handler
            document.getElementById('loginForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                try {
                    const response = await fetch('/api/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        localStorage.setItem('access_token', result.access_token);
                        window.location.href = '/dashboard';
                    } else {
                        showMessage(result.detail || 'Login failed');
                    }
                } catch (error) {
                    showMessage('Network error. Please try again.');
                }
            });
            
            // Register form handler
            document.getElementById('registerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                try {
                    const response = await fetch('/api/register', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        showMessage('Account created successfully! Please login.', 'success');
                        showLogin();
                    } else {
                        showMessage(result.detail || 'Registration failed');
                    }
                } catch (error) {
                    showMessage('Network error. Please try again.');
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RideShare Pro - Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: #f8f9fa;
                color: #333;
            }
            
            .navbar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px 0;
                box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            }
            
            .nav-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0 20px;
            }
            
            .logo {
                color: white;
                font-size: 1.5rem;
                font-weight: 700;
            }
            
            .user-info {
                display: flex;
                align-items: center;
                gap: 15px;
                color: white;
            }
            
            .logout-btn {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            
            .logout-btn:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            
            .dashboard-container {
                max-width: 1200px;
                margin: 40px auto;
                padding: 0 20px;
            }
            
            .welcome-section {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
            }
            
            .welcome-section h1 {
                font-size: 2rem;
                margin-bottom: 10px;
                color: #333;
            }
            
            .welcome-section p {
                color: #666;
                font-size: 1.1rem;
            }
            
            .actions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .action-card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
                cursor: pointer;
                transition: all 0.3s ease;
                border: 2px solid transparent;
            }
            
            .action-card:hover {
                transform: translateY(-5px);
                border-color: #667eea;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
            }
            
            .action-card .icon {
                font-size: 3rem;
                margin-bottom: 15px;
            }
            
            .action-card h3 {
                font-size: 1.5rem;
                margin-bottom: 10px;
                color: #333;
            }
            
            .action-card p {
                color: #666;
                line-height: 1.5;
            }
            
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(5px);
                z-index: 1000;
            }
            
            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 30px;
                border-radius: 20px;
                width: 90%;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            }
            
            .modal h2 {
                margin-bottom: 20px;
                color: #333;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
            }
            
            .form-group input,
            .form-group select,
            .form-group textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .form-group input:focus,
            .form-group select:focus,
            .form-group textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-right: 10px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .btn-secondary {
                background: #6c757d;
            }
            
            .btn-secondary:hover {
                box-shadow: 0 5px 15px rgba(108, 117, 125, 0.3);
            }
            
            .close-btn {
                position: absolute;
                top: 15px;
                right: 20px;
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #999;
            }
            
            .close-btn:hover {
                color: #333;
            }
            
            .history-item {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
            }
            
            .history-item h4 {
                margin-bottom: 5px;
                color: #333;
            }
            
            .history-item p {
                color: #666;
                font-size: 14px;
                margin-bottom: 3px;
            }
            
            .status {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
            }
            
            .status.pending { background: #fff3cd; color: #856404; }
            .status.completed { background: #d4edda; color: #155724; }
            .status.cancelled { background: #f8d7da; color: #721c24; }
            
            @media (max-width: 768px) {
                .actions-grid {
                    grid-template-columns: 1fr;
                }
                
                .modal-content {
                    width: 95%;
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="nav-content">
                <div class="logo">🚗 RideShare Pro</div>
                <div class="user-info">
                    <span id="userName">Welcome</span>
                    <button class="logout-btn" onclick="logout()">Logout</button>
                </div>
            </div>
        </nav>
        
        <div class="dashboard-container">
            <div class="welcome-section">
                <h1>Dashboard</h1>
                <p>Choose your next adventure or manage your rides</p>
            </div>
            
            <div class="actions-grid">
                <div class="action-card" onclick="openModal('rideModal')">
                    <div class="icon">🚕</div>
                    <h3>Request a Ride</h3>
                    <p>Book an instant ride to your destination with our premium fleet</p>
                </div>
                
                <div class="action-card" onclick="openModal('scheduleModal')">
                    <div class="icon">📅</div>
                    <h3>Schedule Ride</h3>
                    <p>Plan ahead and schedule your ride for a specific date and time</p>
                </div>
                
                <div class="action-card" onclick="showHistory()">
                    <div class="icon">📋</div>
                    <h3>Ride History</h3>
                    <p>View your past rides, receipts, and travel patterns</p>
                </div>
            </div>
        </div>
        
        <!-- Request Ride Modal -->
        <div id="rideModal" class="modal">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal('rideModal')">&times;</button>
                <h2>Request a Ride</h2>
                <form id="rideForm">
                    <div class="form-group">
                        <label for="pickup">Pickup Location</label>
                        <input type="text" id="pickup" name="pickup_location" required 
                               placeholder="Enter pickup address">
                    </div>
                    
                    <div class="form-group">
                        <label for="destination">Destination</label>
                        <input type="text" id="destination" name="destination" required 
                               placeholder="Where are you going?">
                    </div>
                    
                    <div class="form-group">
                        <label for="rideType">Ride Type</label>
                        <select id="rideType" name="ride_type" required>
                            <option value="economy">Economy - Most affordable</option>
                            <option value="comfort">Comfort - Extra space</option>
                            <option value="premium">Premium - Luxury experience</option>
                            <option value="xl">XL - Up to 6 passengers</option>
                        </select>
                    </div>
                    
                    <button type="submit" class="btn">Request Ride</button>
                    <button type="button" class="btn btn-secondary" onclick="closeModal('rideModal')">Cancel</button>
                </form>
            </div>
        </div>
        
        <!-- Schedule Ride Modal -->
        <div id="scheduleModal" class="modal">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal('scheduleModal')">&times;</button>
                <h2>Schedule a Ride</h2>
                <form id="scheduleForm">
                    <div class="form-group">
                        <label for="schedulePickup">Pickup Location</label>
                        <input type="text" id="schedulePickup" name="pickup_location" required 
                               placeholder="Enter pickup address">
                    </div>
                    
                    <div class="form-group">
                        <label for="scheduleDestination">Destination</label>
                        <input type="text" id="scheduleDestination" name="destination" required 
                               placeholder="Where are you going?">
                    </div>
                    
                    <div class="form-group">
                        <label for="scheduleType">Ride Type</label>
                        <select id="scheduleType" name="ride_type" required>
                            <option value="economy">Economy - Most affordable</option>
                            <option value="comfort">Comfort - Extra space</option>
                            <option value="premium">Premium - Luxury experience</option>
                            <option value="xl">XL - Up to 6 passengers</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="scheduleTime">Pickup Date & Time</label>
                        <input type="datetime-local" id="scheduleTime" name="scheduled_time" required>
                    </div>
                    
                    <button type="submit" class="btn">Schedule Ride</button>
                    <button type="button" class="btn btn-secondary" onclick="closeModal('scheduleModal')">Cancel</button>
                </form>
            </div>
        </div>
        
        <!-- History Modal -->
        <div id="historyModal" class="modal">
            <div class="modal-content">
                <button class="close-btn" onclick="closeModal('historyModal')">&times;</button>
                <h2>Ride History</h2>
                <div id="historyContent">
                    <p>Loading...</p>
                </div>
            </div>
        </div>
        
        <script>
            // Check authentication
            const token = localStorage.getItem('access_token');
            if (!token) {
                window.location.href = '/';
            }
            
            // Load user info
            async function loadUserInfo() {
                try {
                    const response = await fetch('/api/user/me', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                    
                    if (response.ok) {
                        const user = await response.json();
                        document.getElementById('userName').textContent = `Hello, ${user.full_name}`;
                    } else {
                        logout();
                    }
                } catch (error) {
                    console.error('Error loading user info:', error);
                    logout();
                }
            }
            
            // Modal functions
            function openModal(modalId) {
                document.getElementById(modalId).style.display = 'block';
            }
            
            function closeModal(modalId) {
                document.getElementById(modalId).style.display = 'none';
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                const modals = document.querySelectorAll('.modal');
                modals.forEach(modal => {
                    if (event.target === modal) {
                        modal.style.display = 'none';
                    }
                });
            }
            
            // Logout function
            function logout() {
                localStorage.removeItem('access_token');
                window.location.href = '/';
            }
            
            // Request ride form handler
            document.getElementById('rideForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                try {
                    const response = await fetch('/api/rides/request', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('Ride requested successfully! A driver will be assigned shortly.');
                        closeModal('rideModal');
                        e.target.reset();
                    } else {
                        alert(result.detail || 'Failed to request ride');
                    }
                } catch (error) {
                    alert('Network error. Please try again.');
                }
            });
            
            // Schedule ride form handler
            document.getElementById('scheduleForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData);
                
                // Convert datetime-local to ISO format
                if (data.scheduled_time) {
                    data.scheduled_time = new Date(data.scheduled_time).toISOString();
                }
                
                try {
                    const response = await fetch('/api/rides/request', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('Ride scheduled successfully! You will receive a confirmation.');
                        closeModal('scheduleModal');
                        e.target.reset();
                    } else {
                        alert(result.detail || 'Failed to schedule ride');
                    }
                } catch (error) {
                    alert('Network error. Please try again.');
                }
            });
            
            // Show ride history
            async function showHistory() {
                openModal('historyModal');
                
                try {
                    const response = await fetch('/api/rides/history', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });
                    
                    if (response.ok) {
                        const rides = await response.json();
                        displayHistory(rides);
                    } else {
                        document.getElementById('historyContent').innerHTML = 
                            '<p>Failed to load ride history.</p>';
                    }
                } catch (error) {
                    document.getElementById('historyContent').innerHTML = 
                        '<p>Error loading ride history.</p>';
                }
            }
            
            // Display ride history
            function displayHistory(rides) {
                const historyContent = document.getElementById('historyContent');
                
                if (rides.length === 0) {
                    historyContent.innerHTML = '<p>No rides found. Book your first ride!</p>';
                    return;
                }
                
                let html = '';
                rides.forEach(ride => {
                    const date = new Date(ride.created_at).toLocaleDateString();
                    const time = new Date(ride.created_at).toLocaleTimeString();
                    
                    html += `
                        <div class="history-item">
                            <h4>${ride.pickup_location} → ${ride.destination}</h4>
                            <p><strong>Type:</strong> ${ride.ride_type.charAt(0).toUpperCase() + ride.ride_type.slice(1)}</p>
                            <p><strong>Date:</strong> ${date} at ${time}</p>
                            <p><strong>Status:</strong> <span class="status ${ride.status}">${ride.status}</span></p>
                        </div>
                    `;
                });
                
                historyContent.innerHTML = html;
            }
            
            // Set minimum datetime for scheduling (current time + 30 minutes)
            const now = new Date();
            now.setMinutes(now.getMinutes() + 30);
            document.getElementById('scheduleTime').min = now.toISOString().slice(0, 16);
            
            // Load user info on page load
            loadUserInfo();
        </script>
    </body>
    </html>
    """

# Initialize database on startup


@app.on_event("startup")
async def startup_event():
    init_db()

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
