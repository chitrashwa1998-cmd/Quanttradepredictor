"""
Configuration management for TribexAlpha backend
Environment variables and application settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/tribex_alpha")
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "TribexAlpha Trading Platform"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://0.0.0.0:3000",
        "http://0.0.0.0:5173"
    ]
    
    # External API keys
    UPSTOX_API_KEY: Optional[str] = os.getenv("UPSTOX_API_KEY")
    UPSTOX_API_SECRET: Optional[str] = os.getenv("UPSTOX_API_SECRET")
    UPSTOX_ACCESS_TOKEN: Optional[str] = os.getenv("UPSTOX_ACCESS_TOKEN")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Frontend URL for CORS
    FRONTEND_URL: Optional[str] = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()