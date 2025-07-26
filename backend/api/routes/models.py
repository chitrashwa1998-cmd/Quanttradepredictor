"""Apply the changes to fix the feature calculation and incorporate the complete volatility feature calculation pipeline."""
"""Model management API endpoints Handles model training, loading, and management operations"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import io
import logging

from core.model_manager import ModelManager
from core.database import get_database_dependency

logger = logging.getLogger(__name__)
router = APIRouter()

class TrainingRequest(BaseModel):
    model_name: str
    dataset_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class TrainingResponse(BaseModel):
    success: bool
    model_name: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    db = Depends(get_database_dependency)
):
    """Train a specific model with data from database or parameters"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager

        # Get training data
        if request.dataset_name:
            training_data = db.get_dataset(request.dataset_name)
        else:
            # Use latest training dataset
            datasets = db.list_datasets()
            if not datasets:
                raise ValueError("No training data available")
            training_data = db.get_dataset(datasets[0]['name'])

        # Train model
        results = await model_manager.train_model(request.model_name, training_data)

        return TrainingResponse(
            success=True,
            model_name=request.model_name,
            results=results
        )

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return TrainingResponse(
            success=False,
            model_name=request.model_name,
            error=str(e)
        )

@router.post("/upload-data")
async def upload_training_data(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None,
    db = Depends(get_database_dependency)
):
    """Upload training data from CSV file"""
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate OHLC format
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Use filename as dataset name if not provided
        if not dataset_name:
            dataset_name = (file.filename or "uploaded_data").replace('.csv', '')

        # Store in database
        db.store_dataset(dataset_name, df)

        return {
            "success": True,
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df['Date'].min(),
                "end": df['Date'].max()
            }
        }

    except Exception as e:
        logger.error(f"Data upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
async def get_models_status():
    """Get status of all models"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager

        status = await model_manager.get_model_status()

        return {
            "success": True,
            "models": status.get("models", {}),
            "initialized": status.get("initialized", False)
        }

    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_models():
    """List all available models and their status"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager

        status = await model_manager.get_model_status()

        return {
            "success": True,
            "models": status.get("models", {}),
            "initialized": status.get("initialized", False)
        }

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_name}")
async def delete_model(
    model_name: str,
    db = Depends(get_database_dependency)
):
    """Delete a specific model from database"""
    try:
        # Delete model from database
        db.delete_model(model_name)

        return {
            "success": True,
            "message": f"Model {model_name} deleted successfully"
        }

    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calculate-features")
async def calculate_features(
    request: dict,
    db = Depends(get_database_dependency)
):
    """Calculate features for a specific model type"""
    try:
        dataset_name = request.get('dataset_name')
        model_type = request.get('model_type')

        if not dataset_name or not model_type:
            raise HTTPException(status_code=400, detail="dataset_name and model_type are required")

        # Load dataset
        dataset = db.get_dataset(dataset_name)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        # Calculate features based on model type
        if model_type == "volatility":
            from features.technical_indicators import TechnicalIndicators
            features = TechnicalIndicators.calculate_all_indicators(dataset)

        elif model_type == "direction":
            from features.direction_technical_indicators import DirectionTechnicalIndicators
            features = DirectionTechnicalIndicators.calculate_all_direction_indicators(dataset)

        elif model_type == "profit_probability":
            from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
            features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(dataset)

        elif model_type == "reversal":
            from features.reversal_technical_indicators import ReversalTechnicalIndicators
            features = ReversalTechnicalIndicators.calculate_all_reversal_indicators(dataset)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        return {
            "success": True,
            "model_type": model_type,
            "feature_count": len(features.columns),
            "message": f"Features calculated successfully for {model_type} model"
        }

    except Exception as e:
        logger.error(f"Feature calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model(
    request: dict,
    db = Depends(get_database_dependency)
):
    """Train a specific model"""
    try:
        model_type = request.get('model_type')
        dataset_name = request.get('dataset_name')
        config = request.get('config', {})

        if not model_type or not dataset_name:
            raise HTTPException(status_code=400, detail="Missing model_type or dataset_name")

        # Get dataset
        dataset = db.get_dataset(dataset_name)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        # Train the model
        if model_type == 'volatility':
            from models.volatility_model import VolatilityModel
            model = VolatilityModel()
            result = model.train(dataset)
        elif model_type == 'direction':
            from models.direction_model import DirectionModel
            model = DirectionModel()
            result = model.train(dataset)
        elif model_type == 'profit_probability':
            from models.profit_probability_model import ProfitProbabilityModel
            model = ProfitProbabilityModel()
            result = model.train(dataset)
        elif model_type == 'reversal':
            from models.reversal_model import ReversalModel
            model = ReversalModel()
            result = model.train(dataset)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

        if result and result.get('success'):
            return {
                "success": True,
                "model_type": model_type,
                "dataset_name": dataset_name,
                "mse": result.get('mse'),
                "r2_score": result.get('r2_score'),
                "feature_count": result.get('feature_count'),
                "training_samples": result.get('training_samples')
            }
        else:
            raise HTTPException(status_code=500, detail="Model training failed")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{model_name}/info")
async def get_model_info(
    model_name: str,
    db = Depends(get_database_dependency)
):
    """Get detailed information about a specific model"""
    try:
        from main import app
        model_manager: ModelManager = app.state.model_manager

        model = model_manager.get_model(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        info = {
            "name": model_name,
            "loaded": hasattr(model, 'model') and model.model is not None,
            "features": getattr(model, 'feature_names', []),
            "last_trained": getattr(model, 'last_trained', None),
            "model_type": type(model).__name__
        }

        return {"success": True, "info": info}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))