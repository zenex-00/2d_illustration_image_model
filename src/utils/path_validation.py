"""Path validation utilities to prevent path traversal attacks"""

from pathlib import Path
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_path_within_directory(
    file_path: Path,
    allowed_directory: Path,
    must_exist: bool = False
) -> Path:
    """
    Validate that a file path is within an allowed directory (prevents path traversal).
    
    Args:
        file_path: Path to validate (can be relative or absolute)
        allowed_directory: Directory that the path must be within
        must_exist: If True, file must exist and be a file
    
    Returns:
        Resolved absolute path if valid
    
    Raises:
        ValueError: If path is outside allowed directory
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    # Resolve both paths to absolute
    try:
        resolved_path = file_path.resolve()
        resolved_allowed = allowed_directory.resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {str(e)}")
    
    # Ensure resolved path is within allowed directory
    try:
        # Check if resolved_path is a subpath of resolved_allowed
        resolved_path.relative_to(resolved_allowed)
    except ValueError:
        # Path is outside allowed directory
        raise ValueError(
            f"Path traversal detected: {resolved_path} is outside allowed directory {resolved_allowed}"
        )
    
    # Check if file exists and is a file (if required)
    if must_exist:
        if not resolved_path.exists():
            raise FileNotFoundError(f"File does not exist: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Path is not a file: {resolved_path}")
    
    return resolved_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename (only alphanumeric, dots, dashes, underscores)
    
    Raises:
        ValueError: If filename is invalid
    """
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")
    
    # Remove any path components
    filename = Path(filename).name
    
    # Check for valid characters only
    if not all(c.isalnum() or c in ('.', '-', '_') for c in filename):
        raise ValueError(f"Invalid filename characters: {filename}")
    
    # Check for path traversal attempts
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValueError(f"Path traversal detected in filename: {filename}")
    
    return filename


def validate_intermediate_dir(intermediate_dir: str, base_dir: Optional[Path] = None) -> Path:
    """
    Validate and create intermediate directory safely.
    
    Args:
        intermediate_dir: Path to intermediate directory
        base_dir: Optional base directory to restrict to
    
    Returns:
        Validated Path object
    
    Raises:
        ValueError: If path is invalid or outside base_dir
    """
    intermediate_path = Path(intermediate_dir)
    
    # If base_dir is provided, ensure intermediate_dir is within it
    if base_dir:
        try:
            intermediate_path = validate_path_within_directory(intermediate_path, base_dir)
        except ValueError:
            raise ValueError(
                f"Intermediate directory must be within base directory: {base_dir}"
            )
    
    # Create directory if it doesn't exist
    intermediate_path.mkdir(parents=True, exist_ok=True)
    
    # Verify it's actually a directory
    if not intermediate_path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {intermediate_path}")
    
    return intermediate_path

