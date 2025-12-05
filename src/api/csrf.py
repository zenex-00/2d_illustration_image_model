"""Simple CSRF protection for FastAPI forms"""

import secrets
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class CSRFMiddleware(BaseHTTPMiddleware):
    """Simple CSRF token middleware"""
    
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or secrets.token_urlsafe(32)
    
    async def dispatch(self, request: Request, call_next):
        # Skip CSRF for GET requests and API endpoints
        if request.method == "GET" or request.url.path.startswith("/api/"):
            return await call_next(request)
        
        # For POST requests to UI endpoints, check CSRF token
        if request.method == "POST" and request.url.path.startswith("/ui/"):
            # Get CSRF token from form or header
            form = await request.form()
            csrf_token = form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            
            # Get expected token from session/cookie
            session_token = request.cookies.get("csrf_token")
            
            if not csrf_token or not session_token or csrf_token != session_token:
                raise HTTPException(
                    status_code=403,
                    detail="CSRF token validation failed"
                )
        
        response = await call_next(request)
        
        # Set CSRF token cookie if not present
        if not request.cookies.get("csrf_token"):
            response.set_cookie(
                key="csrf_token",
                value=secrets.token_urlsafe(32),
                httponly=True,
                samesite="lax"
            )
        
        return response


def get_csrf_token(request: Request) -> str:
    """Get CSRF token from request cookie"""
    return request.cookies.get("csrf_token", "")



