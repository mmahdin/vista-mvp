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
from app.routes.clustering_service import (
    get_clustering_service, 
    start_clustering_service, 
    stop_clustering_service,
    ClusteringService
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        start_clustering_service()
        clustering_service = get_clustering_service()
        yield
    finally:
        stop_clustering_service()
        db.close()

app = FastAPI(title="ViSta",  lifespan=lifespan, debug=settings.DEBUG)

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





@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RideShare is running"}
