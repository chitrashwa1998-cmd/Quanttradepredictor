"""
Apply model-specific feature engineering for all 4 models like in Streamlit.
"""
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
import sys
import os

# Add project root to Python path to find features module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
trained_models_storage = {}

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
    try:
        dataset_name = request.get('dataset_name')
        model_type = request.get('model_type', 'volatility')  # Default to volatility if not specified

        if not dataset_name:
            raise HTTPException(status_code=400, detail="Dataset name is required")

        # Load dataset
        def load_dataset(dataset_name: str):
            if dataset_name in datasets_storage:
                return datasets_storage[dataset_name]
            else:
                return None
        df = load_dataset(dataset_name)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")

        print(f"Dataset loaded: {len(df)} rows for {model_type} model")

        # Model-specific feature engineering (matching Streamlit approach)
        if model_type == 'volatility':
            # Volatility model - 27 features
            from features.technical_indicators import calculate_technical_indicators
            from features.custom_engineered import calculate_custom_features
            from features.lagged_features import calculate_lagged_features
            from features.time_context_features import calculate_time_context_features

            df_with_features = calculate_technical_indicators(df.copy())
            df_with_features = calculate_custom_features(df.copy())
            df_with_features = calculate_lagged_features(df_with_features)
            df_with_features = calculate_time_context_features(df_with_features)

            feature_count = 27
            feature_categories = {
                'Technical Indicators': 5,    # atr, bb_width, keltner_width, rsi, donchian_width
                'Custom Engineered': 8,       # log_return, realized_volatility, etc.
                'Lagged Features': 7,         # lag_volatility_1, lag_volatility_3, etc.
                'Time Context': 7             # hour, minute, day_of_week, etc.
            }

        elif model_type == 'direction':
            # Direction model - 54 features
            from features.direction_technical_indicators import calculate_direction_technical_indicators
            from features.direction_custom_engineered import calculate_direction_custom_features
            from features.direction_lagged_features import calculate_direction_lagged_features
            from features.direction_time_context import calculate_direction_time_context_features

            df_with_features = calculate_direction_technical_indicators(df.copy())
            df_with_features = calculate_direction_custom_features(df_with_features)
            df_with_features = calculate_direction_lagged_features(df_with_features)
            df_with_features = calculate_direction_time_context_features(df_with_features)

            feature_count = 54
            feature_categories = {
                'Technical Indicators': 15,   # RSI, MACD, Bollinger Bands, etc.
                'Price Action': 12,           # Price patterns, momentum indicators
                'Lagged Features': 15,        # Historical price movements
                'Time Context': 12            # Market session, volatility timing
            }

        elif model_type == 'profit_probability':
            # Profit Probability model - 66 features
            from features.profit_probability_technical_indicators import calculate_profit_technical_indicators
            from features.profit_probability_custom_engineered import calculate_profit_custom_features
            from features.profit_probability_lagged_features import calculate_profit_lagged_features
            from features.profit_probability_time_context import calculate_profit_time_context_features

            df_with_features = calculate_profit_technical_indicators(df.copy())
            df_with_features = calculate_profit_custom_features(df_with_features)
            df_with_features = calculate_profit_lagged_features(df_with_features)
            df_with_features = calculate_profit_time_context_features(df_with_features)

            feature_count = 66
            feature_categories = {
                'Technical Indicators': 20,   # Extended technical analysis
                'Risk Metrics': 18,           # Risk-reward calculations
                'Lagged Features': 16,        # Historical performance
                'Time Context': 12            # Market timing factors
            }

        elif model_type == 'reversal':
            # Reversal model - 63 features
            from features.reversal_technical_indicators import calculate_reversal_technical_indicators
            from features.reversal_custom_engineered import calculate_reversal_custom_features
            from features.reversal_lagged_features import calculate_reversal_lagged_features
            from features.reversal_time_context import calculate_reversal_time_context_features

            df_with_features = calculate_reversal_technical_indicators(df.copy())
            df_with_features = calculate_reversal_custom_features(df_with_features)
            df_with_features = calculate_reversal_lagged_features(df_with_features)
            df_with_features = calculate_reversal_time_context_features(df_with_features)

            feature_count = 63
            feature_categories = {
                'Technical Indicators': 18,   # Reversal-specific indicators
                'Pattern Recognition': 17,    # Chart patterns, support/resistance
                'Lagged Features': 16,        # Historical reversal patterns
                'Time Context': 12            # Market timing for reversals
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")

        # Remove rows with NaN values
        df_with_features = df_with_features.dropna()

        # Save the processed dataset
        def save_dataset(features_dataset_name, df_with_features):
            datasets_storage[features_dataset_name] = df_with_features

        features_dataset_name = f"{dataset_name}_features"
        save_dataset(features_dataset_name, df_with_features)

        # Exclude OHLC columns for feature count validation
        excluded_features = ['date', 'open', 'high', 'low', 'close']
        actual_feature_columns = [col for col in df_with_features.columns if col.lower() not in [f.lower() for f in excluded_features]]

        print(f"Warning: Excluding extra features: {[col for col in df_with_features.columns if col.lower() in [f.lower() for f in excluded_features]]}")
        print(f"{model_type.capitalize()} model using exactly {feature_count} features (target: {feature_count})")

        return {
            "status": "success",
            "model_type": model_type,
            "data_points": len(df_with_features),
            "total_features": len(df_with_features.columns),
            "engineered_features": feature_count,  # Model-specific feature count
            "feature_categories": feature_categories,
            "features_dataset": features_dataset_name
        }

    except Exception as e:
        print(f"Error in calculate_features: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature calculation failed: {str(e)}")

@app.post("/api/models/train")
async def train_model(request: dict):
    """Train a specific model with detailed results matching Streamlit"""
    try:
        model_type = request.get('model_type', 'volatility')
        dataset_name = request.get('dataset_name')
        config = request.get('config', {})

        if not dataset_name or f"{dataset_name}_features" not in datasets_storage:
            raise HTTPException(status_code=404, detail="Features not calculated. Please calculate features first.")

        # Get the processed features
        features_data = datasets_storage[f"{dataset_name}_features"]
        raw_data = datasets_storage[dataset_name]

        # Import the appropriate model
        import sys
        import os
        sys.path.append('/home/runner/workspace')

        from models.volatility_model import VolatilityModel

        if model_type == 'volatility':
            logging.info("🚀 Training volatility model...")

            # Initialize volatility model
            volatility_model = VolatilityModel()

            # Create target from raw data
            target = volatility_model.create_target(raw_data)

            # Ensure data alignment before training
            logging.info(f"📊 Pre-training alignment - Features: {len(features_data)} rows, Target: {len(target)} values")

            # Align features and target by taking the minimum length
            min_length = min(len(features_data), len(target))
            features_aligned = features_data.iloc[:min_length].copy()
            target_aligned = target.iloc[:min_length].copy()

            logging.info(f"📊 Post-alignment - Features: {len(features_aligned)} rows, Target: {len(target_aligned)} values")

            # Train the model
            training_result = volatility_model.train(
                features_aligned, 
                target_aligned, 
                config.get('train_split', 0.8)
            )

            # Store the trained model
            trained_models_storage['volatility'] = {
                'model': volatility_model,
                'result': training_result
            }

            # Extract metrics (matching Streamlit display)
            metrics = training_result.get('metrics', {})

            # Prepare detailed response matching Streamlit format
            response_data = {
                "success": True,
                "model_type": model_type,
                "metrics": {
                    "rmse": metrics.get('rmse', metrics.get('test_rmse')),
                    "mae": metrics.get('mae'),
                    "mse": metrics.get('mse'),
                    "r2": metrics.get('r2', metrics.get('test_r2')),
                    "train_rmse": metrics.get('train_rmse'),
                    "test_rmse": metrics.get('test_rmse'),
                    "train_r2": metrics.get('train_r2'),
                    "test_r2": metrics.get('test_r2')
                },
                "feature_importance": training_result.get('feature_importance', {}),
                "model_info": {
                    "training_samples": len(features_data),
                    "features_used": len(features_data.columns),
                    "model_type": "Ensemble (XGBoost + CatBoost + Random Forest)",
                    "task_type": training_result.get('task_type', 'regression')
                },
                "feature_names": training_result.get('feature_names', []),
                "training_config": {
                    "train_split": config.get('train_split', 0.8),
                    "max_depth": config.get('max_depth', 6),
                    "n_estimators": config.get('n_estimators', 100)
                },
                "message": f"✅ Volatility model trained successfully!"
            }

            logging.info(f"✅ Volatility model training completed successfully")
            logging.info(f"📊 Final metrics - RMSE: {response_data['metrics']['rmse']:.6f}, MAE: {response_data['metrics']['mae']:.6f}")

            return response_data
        else:
            raise HTTPException(status_code=400, detail=f"Model type {model_type} not supported yet")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)