"""Security utilities for API authentication"""

import os
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# API Key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> Optional[str]:
    """
    Validate API key if configured
    
    Returns:
        The API key if valid, or None if no key configured
        
    Raises:
        HTTPException: If API key is invalid
    """
    expected_key = os.getenv("FASTAPI_API_KEY")
    
    # If no key configured, allow open access
    if not expected_key:
        return None
        
    if api_key_header == expected_key:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )
