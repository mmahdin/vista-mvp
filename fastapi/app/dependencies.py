from fastapi import Depends, HTTPException, Request, status
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from typing import Annotated, Optional

from app.config import settings
from app.database.base import get_db
from app.database.models import User

ACCESS_TOKEN_COOKIE_NAME = "access_token"


def get_current_user(
    request: Request,
    db: Annotated[Session, Depends(get_db)]
) -> Optional[User]:
    """
    Retrieves the current user based on the JWT stored in cookies.
    Returns None if no token or invalid token is found.
    """
    token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if not token:
        return None

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        email: Optional[str] = payload.get("sub")
        if not email:
            return None
    except JWTError:
        return None

    return db.query(User).filter(User.email == email).first()


def protected_route(
    user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Ensures that a valid user is present, or raises HTTP 401 Unauthorized.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"Location": "/login"},
        )
    return user
