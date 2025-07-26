"""
Simplified FastAPI main with basic API endpoints
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import io
import logging
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

# In-memory storage for datasets (for demo purposes)
datasets_storage = {}

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
            "total_datasets": len(datasets_storage),
            "total_records": sum(len(df) for df in datasets_storage.values()),
            "total_models": 0,
            "total_trained_models": 0,
            "total_predictions": 0,
            "datasets": list(datasets_storage.keys()),
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
    result = []
    for name, df in datasets_storage.items():
        dataset_info = {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "purpose": "training"
        }

        # Add date range if Date column exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                dataset_info["date_range"] = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            except:
                dataset_info["date_range"] = "Unknown"

        result.append(dataset_info)

    return result

@app.get("/api/models/list")
async def list_models():
    return {
        "models": ["volatility", "direction", "profit_probability", "reversal"]
    }

@app.post("/api/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    purpose: Optional[str] = Form("training")
):
    """Upload CSV data file"""
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Basic validation
        if df.empty:
            raise ValueError("Uploaded file is empty")

        # Use filename as dataset name if not provided
        if not dataset_name:
            dataset_name = (file.filename or "uploaded_data").replace('.csv', '')

        # Store in memory (in a real app, this would go to database)
        datasets_storage[dataset_name] = df

        # Create preview data
        preview_data = None
        if len(df) > 0:
            preview_data = {
                "columns": list(df.columns),
                "rows": df.head(5).values.tolist()
            }

        # Get date range
        date_range = "Unknown"
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            except:
                pass

        return {
            "success": True,
            "dataset_name": dataset_name,
            "records_processed": len(df),
            "date_range": date_range,
            "columns": list(df.columns),
            "preview": preview_data,
            "message": f"Successfully uploaded {len(df)} rows"
        }

    except Exception as e:
        logging.error(f"Data upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/datasets/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get a specific dataset"""
    try:
        if dataset_name not in datasets_storage:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        df = datasets_storage[dataset_name]
        
        return {
            "success": True,
            "dataset_name": dataset_name,
            "data": df.to_dict('records'),
            "rows": len(df),
            "columns": list(df.columns)
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/data/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a specific dataset"""
    try:
        if dataset_name not in datasets_storage:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        del datasets_storage[dataset_name]

        return {
            "success": True,
            "message": f"Dataset {dataset_name} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete dataset {dataset_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/calculate-features")
async def calculate_features(request: dict):
    """Calculate technical indicators for volatility model exactly as in Streamlit"""
    try:
        dataset_name = request.get('dataset_name')
        if not dataset_name or dataset_name not in datasets_storage:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get the raw dataset
        raw_data = datasets_storage[dataset_name]
        
        # Import the exact same modules used in Streamlit
        import sys
        import os
        sys.path.append('/home/runner/workspace')
        
        from features.technical_indicators import TechnicalIndicators
        from utils.data_processing import DataProcessor
        
        # Calculate features using the exact same process as Streamlit
        logging.info("🔧 Calculating volatility-specific technical indicators...")
        
        # Step 1: Calculate all indicators (same as Streamlit)
        features_data = TechnicalIndicators.calculate_all_indicators(raw_data)
        
        # Step 2: Clean the data (same as Streamlit)
        features_clean = DataProcessor.clean_data(features_data)
        
        # Store the calculated features
        datasets_storage[f"{dataset_name}_features"] = features_clean
        
        # Count engineered features (excluding OHLC columns)
        ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in features_clean.columns if col not in ohlc_cols]
        
        logging.info(f"✅ Volatility features calculated: {len(feature_cols)} engineered features, {len(features_clean)} data points")
        
        return {
            "success": True,
            "message": "Volatility technical indicators calculated successfully!",
            "total_features": len(features_clean.columns),
            "data_points": len(features_clean),
            "engineered_features": len(feature_cols),
            "feature_columns": feature_cols,
            "sample_data": features_clean.head(10).to_dict('records') if len(features_clean) > 0 else []
        }
        
    except Exception as e:
        logging.error(f"Error calculating volatility features: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Feature calculation failed: {str(e)}")

@app.post("/api/models/train")
async def train_model(request: dict):
    """Train a specific model"""
    try:
        model_type = request.get('model_type')
        dataset_name = request.get('dataset_name')
        
        if not model_type:
            raise HTTPException(status_code=400, detail="model_type is required")
        if not dataset_name or dataset_name not in datasets_storage:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Mock training results
        mock_metrics = {
            'volatility': {'rmse': 0.0245, 'mae': 0.0198, 'mse': 0.0006},
            'direction': {'accuracy': 0.67, 'precision': 0.65, 'recall': 0.69},
            'profit_probability': {'accuracy': 0.72, 'precision': 0.70, 'recall': 0.74},
            'reversal': {'accuracy': 0.63, 'precision': 0.61, 'recall': 0.66}
        }
        
        return {
            "success": True,
            "model_type": model_type,
            "metrics": mock_metrics.get(model_type, {}),
            "message": f"{model_type.capitalize()} model trained successfully (mock)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)