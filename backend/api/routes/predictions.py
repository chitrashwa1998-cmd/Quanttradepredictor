"""
Predictions API endpoints
Handles ML model predictions and real-time inference
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

from core.model_manager import ModelManager
from core.database import get_database_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictionRequest(BaseModel):
    model_name: str
    data: Dict[str, Any]
    
class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    model_name: str
    timestamp: str
    error: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest,
    db = Depends(get_database_dependency)
):
    """Generate prediction using specified model"""
    try:
        # Get model manager from app state
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        # Generate prediction
        result = await model_manager.predict(request.model_name, request.data)
        
        return PredictionResponse(
            success=True,
            prediction=result,
            model_name=request.model_name,
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return PredictionResponse(
            success=False,
            model_name=request.model_name,
            timestamp=pd.Timestamp.now().isoformat(),
            error=str(e)
        )

@router.get("/models/status")
async def get_models_status():
    """Get status of all available models"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        status = await model_manager.get_model_status()
        return {"success": True, "status": status}
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/live")
async def get_live_predictions(
    db = Depends(get_database_dependency)
):
    """Get latest live predictions from database"""
    try:
        # Get recent predictions from database
        predictions = db.get_recent_predictions(limit=10)
        
        return {
            "success": True,
            "predictions": predictions,
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get live predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def batch_predictions(
    requests: List[PredictionRequest],
    db = Depends(get_database_dependency)
):
    """Generate batch predictions for multiple models/datasets"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager
        
        results = []
        for request in requests:
            try:
                result = await model_manager.predict(request.model_name, request.data)
                results.append({
                    "success": True,
                    "prediction": result,
                    "model_name": request.model_name
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "model_name": request.model_name,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total": len(requests)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))