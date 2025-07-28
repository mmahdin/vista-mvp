from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from .base import Base
from datetime import datetime, timezone
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    phone_number = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Add relationship to locations
    locations = relationship(
        "Location", back_populates="user", cascade="all, delete-orphan")


class Ride(Base):
    __tablename__ = "rides"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_location = Column(String)
    end_location = Column(String)
    ride_type = Column(String, default="standard")
    time = Column(DateTime, default=datetime.now(timezone.utc))
    status = Column(String, default="requested")
    driver_id = Column(Integer, nullable=True)
    price = Column(Integer, nullable=True)


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"),
                     nullable=False)  # Add foreign key

    origin_lat = Column(Float, nullable=False)
    origin_lng = Column(Float, nullable=False)

    destination_lat = Column(Float, nullable=False)
    destination_lng = Column(Float, nullable=False)

    stored_at = Column(DateTime, default=datetime.now(timezone.utc))

    # Add relationship to user
    user = relationship("User", back_populates="locations")
