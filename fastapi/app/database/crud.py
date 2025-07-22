from sqlalchemy.orm import Session
# Import from security instead of auth.utils
from app.security import get_password_hash
from .models import User, Ride


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, email: str, password: str, full_name: str):
    hashed_password = get_password_hash(password)
    db_user = User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_rides(db: Session, user_id: int):
    return db.query(Ride).filter(Ride.user_id == user_id).order_by(Ride.time.desc()).all()


def create_ride(db: Session, ride_data: dict):
    db_ride = Ride(**ride_data)
    db.add(db_ride)
    db.commit()
    db.refresh(db_ride)
    return db_ride
