from pydantic import BaseModel, EmailStr
from datetime import datetime


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str
    full_name: str


class User(UserBase):
    id: int
    full_name: str
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True


class RideCreate(BaseModel):
    start_location: str
    end_location: str
    ride_type: str = "standard"


class Ride(RideCreate):
    id: int
    user_id: int
    status: str
    time: datetime
    price: int | None = None

    class Config:
        orm_mode = True
