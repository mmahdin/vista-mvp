from fastapi import Depends, HTTPException, Request, status
from jose import jwt
from fastapi.templating import Jinja2Templates
from app.config import settings
from app.database.base import get_db
from app.database.models import User
from sqlalchemy.orm import Session


def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
):
    token = request.cookies.get("access_token")
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.SECRET_KEY,
                             algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if email is None:
            return None
    except jwt.JWTError:
        return None

    user = db.query(User).filter(User.email == email).first()
    return user


def protected_route(user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"Location": "/login"},
        )
    return user
