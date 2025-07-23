from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database.base import Base, engine, SessionLocal, get_db
from app.routes import home, ride, schedule, history
from app.auth.router import router as auth_router
from app.config import settings
from sqlalchemy.orm import Session
from app.database.crud import get_user_by_email, create_user
from contextlib import asynccontextmanager


app = FastAPI(title="ViSta", debug=settings.DEBUG)

# Setup database
Base.metadata.create_all(bind=engine)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(auth_router)
app.include_router(home.router)
app.include_router(ride.router)
app.include_router(schedule.router)
app.include_router(history.router)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        if not get_user_by_email(db, "admin@example.com"):
            create_user(
                db,
                email="admin@example.com",
                password="securepassword",
                full_name="Admin User"
            )
        yield
    finally:
        db.close()


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RideShare is running"}
