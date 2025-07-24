"""
FastAPI Backend for TribexAlpha Trading Platform
Main application entry point with CORS, middleware, and route registration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import sys
import logging

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.routes import predictions, models, data, websocket, live_data
from core.config import settings
from core.database import get_database
from core.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting TribexAlpha backend...")
    
    # Initialize database
    try:
        db = get_database()
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Initialize model manager
    try:
        model_manager = ModelManager()
        await model_manager.initialize()
        app.state.model_manager = model_manager
        logger.info("✅ Model manager initialized")
    except Exception as e:
        logger.error(f"❌ Model manager initialization failed: {e}")
        raise
    
    logger.info("🚀 TribexAlpha backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TribexAlpha backend...")

app = FastAPI(
    title="TribexAlpha Trading Platform API",
    description="Advanced quantitative trading platform with ML-powered predictions",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://0.0.0.0:3000",
        "http://0.0.0.0:5173",
        getattr(settings, 'FRONTEND_URL', "*")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(websocket.router, prefix="/api/ws", tags=["websocket"])
app.include_router(live_data.router, prefix="/api/live-data", tags=["live-data"])

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "TribexAlpha Backend",
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TribexAlpha Trading Platform API",
        "version": "2.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )