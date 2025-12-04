"""Secrets management for API keys and tokens"""

import os
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class SecretsManager:
    """Manages secrets from environment variables and .env files"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize secrets manager.
        
        Args:
            env_file: Optional path to .env file (defaults to .env in project root)
        """
        # Load .env file if available
        if DOTENV_AVAILABLE:
            if env_file:
                load_dotenv(env_file)
            else:
                # Try to find .env in project root
                project_root = Path(__file__).parent.parent.parent
                env_path = project_root / ".env"
                if env_path.exists():
                    load_dotenv(env_path)
    
    def get(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Get secret value from environment.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: If True, raise error if secret is missing
            
        Returns:
            Secret value or default
            
        Raises:
            ValueError: If required secret is missing
        """
        value = os.getenv(key, default)
        
        if required and value is None:
            raise ValueError(f"Required secret '{key}' is not set in environment variables")
        
        return value
    
    def get_api_key(self, service: str = "default") -> Optional[str]:
        """
        Get API key for a service.
        
        Args:
            service: Service name (e.g., 'huggingface', 'openai')
            
        Returns:
            API key or None
        """
        key_name = f"{service.upper()}_API_KEY" if service != "default" else "API_KEY"
        return self.get(key_name)
    
    def get_model_token(self, model_provider: str = "huggingface") -> Optional[str]:
        """
        Get model access token.
        
        Args:
            model_provider: Model provider name (e.g., 'huggingface')
            
        Returns:
            Access token or None
        """
        key_name = f"{model_provider.upper()}_TOKEN"
        return self.get(key_name)
    
    def get_required(self, key: str) -> str:
        """
        Get required secret (raises error if missing).
        
        Args:
            key: Environment variable name
            
        Returns:
            Secret value
            
        Raises:
            ValueError: If secret is missing
        """
        value = self.get(key, required=True)
        if value is None:
            raise ValueError(f"Required secret '{key}' is not set")
        return value
    
    def has_secret(self, key: str) -> bool:
        """
        Check if secret exists.
        
        Args:
            key: Environment variable name
            
        Returns:
            True if secret exists
        """
        return os.getenv(key) is not None


# Global instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager






