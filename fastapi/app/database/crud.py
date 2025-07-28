from sqlalchemy.exc import IntegrityError
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy.orm import Session
# Import from security instead of auth.utils
from app.security import get_password_hash
from .models import User, Ride, Location


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


def create_location(
    db: Session,
    user_id: int,
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float
):
    # Check if user already has a location
    existing_location = db.query(Location).filter(
        Location.user_id == user_id).first()

    if existing_location:
        # User already has a location, UPDATE it using the dedicated function
        return update_location_by_user_id(
            db=db,
            user_id=user_id,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_lat=destination_lat,
            destination_lng=destination_lng
        )

    # Create new location entry
    db_location = Location(
        user_id=user_id,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        destination_lat=destination_lat,
        destination_lng=destination_lng,
        stored_at=datetime.now(timezone.utc)
    )

    try:
        db.add(db_location)
        db.commit()
        db.refresh(db_location)
        return db_location
    except IntegrityError:
        db.rollback()
        # In case of race condition, use update function
        return update_location_by_user_id(
            db=db,
            user_id=user_id,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_lat=destination_lat,
            destination_lng=destination_lng
        )


def delete_location_by_user_id(db: Session, user_id: int) -> bool:
    location = db.query(Location).filter(Location.user_id == user_id).first()

    if location:
        db.delete(location)
        db.commit()
        return True

    return False


def get_location_by_user_id(db: Session, user_id: int):
    return db.query(Location).filter(Location.user_id == user_id).first()


def update_location_by_user_id(
    db: Session,
    user_id: int,
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float
):
    location = db.query(Location).filter(Location.user_id == user_id).first()

    if location:
        location.origin_lat = origin_lat
        location.origin_lng = origin_lng
        location.destination_lat = destination_lat
        location.destination_lng = destination_lng
        location.stored_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(location)
        return location

    return None


def get_all_locations_as_dataframe(db: Session) -> pd.DataFrame:
    try:
        # Use pandas to directly read from database
        query = "SELECT * FROM locations ORDER BY stored_at DESC"
        df = pd.read_sql(query, con=db.bind)
        return df

    except Exception as e:
        print(f"Error reading data from database: {str(e)}")
        return pd.DataFrame(columns=[
            'id', 'origin_lat', 'origin_lng',
            'destination_lat', 'destination_lng', 'stored_at'
        ])
