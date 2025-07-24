"""
Simplified FastAPI main with basic API endpoints
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

app = FastAPI(title="TribexAlpha API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "TribexAlpha API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "TribexAlpha API"}

# Mock API endpoints for frontend functionality
@app.get("/api/data/database/info")
async def get_database_info():
    return {
        "info": {
            "database_type": "postgresql_row_based",
            "total_datasets": 0,
            "total_records": 0,
            "total_models": 0,
            "total_trained_models": 0,
            "total_predictions": 0,
            "datasets": [],
            "backend": "PostgreSQL (Row-Based)",
            "storage_type": "Row-Based",
            "supports_append": True,
            "supports_range_queries": True
        }
    }

@app.get("/api/predictions/models/status")
async def get_models_status():
    return {
        "status": {
            "initialized": True,
            "models": {
                "volatility": {"loaded": True, "features": ["close", "volume", "atr", "bb_width"]},
                "direction": {"loaded": True, "features": ["rsi", "macd", "bb_position", "volume_sma"]},
                "profit_probability": {"loaded": True, "features": ["volatility", "momentum", "support_resistance"]},
                "reversal": {"loaded": True, "features": ["divergence", "overbought", "volume_spike"]}
            }
        }
    }

@app.get("/api/data/datasets")
async def list_datasets():
    return []  # Empty list for now

@app.get("/api/models/list")
async def list_models():
    return {
        "models": ["volatility", "direction", "profit_probability", "reversal"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)