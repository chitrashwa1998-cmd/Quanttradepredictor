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

        # Calculate features based on model type
        if model_type == "volatility":
            from features.technical_indicators import TechnicalIndicators
            df_with_features = TechnicalIndicators.calculate_volatility_indicators(df)

            # Add custom volatility features
            from features.custom_engineered import compute_custom_volatility_features
            df_with_features = compute_custom_volatility_features(df_with_features)

            # Add lagged features
            from features.lagged_features import add_volatility_lagged_features
            df_with_features = add_volatility_lagged_features(df_with_features)

            # Add time context features
            from features.time_context_features import add_time_context_features
            df_with_features = add_time_context_features(df_with_features)

        elif model_type == "direction":
            from features.direction_technical_indicators import DirectionTechnicalIndicators
            df_with_features = DirectionTechnicalIndicators.calculate_all_direction_indicators(df)

        elif model_type == "reversal":
            from features.reversal_technical_indicators import ReversalTechnicalIndicators
            df_with_features = ReversalTechnicalIndicators.calculate_all_reversal_indicators(df)
            
            # Skip time context features for reversal model to avoid datetime issues
            print("✅ Reversal features calculated (skipping time context features)")
            # Add time context features only if we have a proper datetime index
            try:
                if hasattr(df_with_features.index, 'hour'):
                    from features.time_context_features import add_time_context_features
                    df_with_features = add_time_context_features(df_with_features)
                    print("✅ Time context features added")
                else:
                    print("⚠️ Skipping time context features - no datetime index")
            except Exception as e:
                print(f"⚠️ Skipping time context features due to error: {str(e)}")

        elif model_type == "profit_probability":
            from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
            df_with_features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(df)

        else:
            # Default to all indicators
            from features.technical_indicators import TechnicalIndicators
            df_with_features = TechnicalIndicators.calculate_all_indicators(df)

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
        
        # Define model-specific feature counts and categories
        model_feature_info = {
            'volatility': {
                'count': 28,
                'categories': ['Technical Indicators', 'Volatility Measures', 'Statistical Features', 'Lagged Features']
            },
            'direction': {
                'count': 54,
                'categories': ['Momentum Indicators', 'Trend Analysis', 'Support/Resistance', 'Volume Analysis']
            },
            'profit_probability': {
                'count': 66,
                'categories': ['Risk Metrics', 'Probability Features', 'Market Microstructure', 'Pattern Recognition']
            },
            'reversal': {
                'count': 63,
                'categories': ['Divergence Indicators', 'Overbought/Oversold', 'Volume Patterns', 'Price Action']
            }
        }
        
        feature_count = len(actual_feature_columns)
        expected_count = model_feature_info.get(model_type, {}).get('count', feature_count)
        feature_categories = model_feature_info.get(model_type, {}).get('categories', ['Technical Features'])

        print(f"Warning: Excluding extra features: {[col for col in df_with_features.columns if col.lower() in [f.lower() for f in excluded_features]]}")
        print(f"{model_type.capitalize()} model calculated {feature_count} features (expected: {expected_count})")

        return {
            "status": "success",
            "model_type": model_type,
            "data_points": len(df_with_features),
            "total_features": len(df_with_features.columns),
            "engineered_features": feature_count,
            "expected_features": expected_count,
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

        # Import the appropriate models
        import sys
        import os
        sys.path.append('/home/runner/workspace')

        from models.volatility_model import VolatilityModel
        from models.direction_model import DirectionModel
        from models.profit_probability_model import ProfitProbabilityModel
        from models.reversal_model import ReversalModel

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

            # Convert numpy values to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj

            # Prepare detailed response matching Streamlit format
            raw_response_data = {
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
            
            # Convert numpy types to Python types for JSON serialization
            response_data = convert_numpy_types(raw_response_data)

            logging.info(f"✅ Volatility model training completed successfully")
            logging.info(f"📊 Final metrics - RMSE: {response_data['metrics']['rmse']:.6f}, MAE: {response_data['metrics']['mae']:.6f}")

            return response_data

        elif model_type == 'direction':
            logging.info("🚀 Training direction model...")

            # Initialize direction model
            direction_model = DirectionModel()

            # Create target from raw data
            target = direction_model.create_target(raw_data)

            # Ensure data alignment before training
            logging.info(f"📊 Pre-training alignment - Features: {len(features_data)} rows, Target: {len(target)} values")

            # Align features and target by taking the minimum length
            min_length = min(len(features_data), len(target))
            features_aligned = features_data.iloc[:min_length].copy()
            target_aligned = target.iloc[:min_length].copy()

            logging.info(f"📊 Post-alignment - Features: {len(features_aligned)} rows, Target: {len(target_aligned)} values")

            # Train the model
            training_result = direction_model.train(
                features_aligned, 
                target_aligned, 
                config.get('train_split', 0.8),
                max_depth=config.get('max_depth', 6),
                n_estimators=config.get('n_estimators', 100)
            )

            # Store the trained model
            trained_models_storage['direction'] = {
                'model': direction_model,
                'result': training_result
            }

            # Extract metrics (matching Streamlit display)
            metrics = training_result.get('metrics', {})

            # Convert numpy values to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj

            # Prepare detailed response matching Streamlit format
            raw_response_data = {
                "success": True,
                "model_type": model_type,
                "metrics": {
                    "accuracy": metrics.get('accuracy', 0),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1": metrics.get('f1', 0),
                    "train_accuracy": metrics.get('train_accuracy', 0),
                    "test_accuracy": metrics.get('test_accuracy', 0)
                },
                "feature_importance": training_result.get('feature_importance', {}),
                "model_info": {
                    "training_samples": len(features_data),
                    "features_used": len(features_data.columns),
                    "model_type": "Ensemble (XGBoost + CatBoost + Random Forest)",
                    "task_type": training_result.get('task_type', 'classification')
                },
                "feature_names": training_result.get('feature_names', []),
                "training_config": {
                    "train_split": config.get('train_split', 0.8),
                    "max_depth": config.get('max_depth', 6),
                    "n_estimators": config.get('n_estimators', 100)
                },
                "message": f"✅ Direction model trained successfully!"
            }
            
            # Convert numpy types to Python types for JSON serialization
            response_data = convert_numpy_types(raw_response_data)

            logging.info(f"✅ Direction model training completed successfully")
            logging.info(f"📊 Final metrics - Accuracy: {response_data['metrics']['accuracy']:.4f}")

            return response_data

        elif model_type == 'profit_probability':
            logging.info("🚀 Training profit probability model...")

            # Initialize profit probability model
            profit_model = ProfitProbabilityModel()

            # Create target from raw data
            target = profit_model.create_target(raw_data)

            # Ensure data alignment before training
            logging.info(f"📊 Pre-training alignment - Features: {len(features_data)} rows, Target: {len(target)} values")

            # Align features and target by taking the minimum length
            min_length = min(len(features_data), len(target))
            features_aligned = features_data.iloc[:min_length].copy()
            target_aligned = target.iloc[:min_length].copy()

            logging.info(f"📊 Post-alignment - Features: {len(features_aligned)} rows, Target: {len(target_aligned)} values")

            # Train the model
            training_result = profit_model.train(
                features_aligned, 
                target_aligned, 
                config.get('train_split', 0.8)
            )

            # Store the trained model
            trained_models_storage['profit_probability'] = {
                'model': profit_model,
                'result': training_result
            }

            # Extract metrics (matching Streamlit display)
            metrics = training_result.get('metrics', {})

            # Convert numpy values to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj

            # Prepare detailed response matching Streamlit format
            raw_response_data = {
                "success": True,
                "model_type": model_type,
                "metrics": {
                    "accuracy": metrics.get('accuracy', 0),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1": metrics.get('f1', 0),
                    "roc_auc": metrics.get('roc_auc', 0),
                    "train_accuracy": metrics.get('train_accuracy', 0),
                    "test_accuracy": metrics.get('test_accuracy', 0)
                },
                "feature_importance": training_result.get('feature_importance', {}),
                "model_info": {
                    "training_samples": len(features_data),
                    "features_used": len(features_data.columns),
                    "model_type": "Ensemble (XGBoost + CatBoost + Random Forest)",
                    "task_type": training_result.get('task_type', 'classification')
                },
                "feature_names": training_result.get('feature_names', []),
                "training_config": {
                    "train_split": config.get('train_split', 0.8),
                    "max_depth": config.get('max_depth', 6),
                    "n_estimators": config.get('n_estimators', 100)
                },
                "message": f"✅ Profit probability model trained successfully!"
            }
            
            # Convert numpy types to Python types for JSON serialization
            response_data = convert_numpy_types(raw_response_data)

            logging.info(f"✅ Profit probability model training completed successfully")
            logging.info(f"📊 Final metrics - Accuracy: {response_data['metrics']['accuracy']:.4f}, ROC AUC: {response_data['metrics']['roc_auc']:.4f}")

            return response_data

        elif model_type == 'reversal':
            logging.info("🚀 Training reversal model...")

            # Initialize reversal model
            reversal_model = ReversalModel()

            # Create target from raw data
            target = reversal_model.create_target(raw_data)

            # Ensure data alignment before training
            logging.info(f"📊 Pre-training alignment - Features: {len(features_data)} rows, Target: {len(target)} values")

            # Align features and target by taking the minimum length
            min_length = min(len(features_data), len(target))
            features_aligned = features_data.iloc[:min_length].copy()
            target_aligned = target.iloc[:min_length].copy()

            logging.info(f"📊 Post-alignment - Features: {len(features_aligned)} rows, Target: {len(target_aligned)} values")

            # Train the model
            training_result = reversal_model.train(
                features_aligned, 
                target_aligned, 
                config.get('train_split', 0.8)
            )

            # Store the trained model
            trained_models_storage['reversal'] = {
                'model': reversal_model,
                'result': training_result
            }

            # Extract metrics (matching Streamlit display)
            metrics = training_result.get('metrics', {})

            # Convert numpy values to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj

            # Prepare detailed response matching Streamlit format
            raw_response_data = {
                "success": True,
                "model_type": model_type,
                "metrics": {
                    "accuracy": metrics.get('accuracy', 0),
                    "precision": metrics.get('precision', 0),
                    "recall": metrics.get('recall', 0),
                    "f1": metrics.get('f1', 0),
                    "roc_auc": metrics.get('roc_auc', 0),
                    "train_accuracy": metrics.get('train_accuracy', 0),
                    "test_accuracy": metrics.get('test_accuracy', 0)
                },
                "feature_importance": training_result.get('feature_importance', {}),
                "model_info": {
                    "training_samples": len(features_data),
                    "features_used": len(features_data.columns),
                    "model_type": "Ensemble (XGBoost + CatBoost + Random Forest)",
                    "task_type": training_result.get('task_type', 'classification')
                },
                "feature_names": training_result.get('feature_names', []),
                "training_config": {
                    "train_split": config.get('train_split', 0.8),
                    "max_depth": config.get('max_depth', 6),
                    "n_estimators": config.get('n_estimators', 100)
                },
                "message": f"✅ Reversal model trained successfully!"
            }
            
            # Convert numpy types to Python types for JSON serialization
            response_data = convert_numpy_types(raw_response_data)

            logging.info(f"✅ Reversal model training completed successfully")
            logging.info(f"📊 Final metrics - Accuracy: {response_data['metrics']['accuracy']:.4f}, ROC AUC: {response_data['metrics']['roc_auc']:.4f}")

            return response_data

        else:
            raise HTTPException(status_code=400, detail=f"Model type {model_type} not supported")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

# === PREDICTIONS ENDPOINTS ===

@app.post("/api/predictions/generate")
async def generate_predictions(request: dict):
    """Generate predictions for a specific model type - exact Streamlit functionality"""
    try:
        model_type = request.get('model_type')
        dataset_name = request.get('dataset_name')
        config = request.get('config', {})
        
        logging.info(f"🔮 Generating {model_type} predictions for dataset: {dataset_name}")
        
        # Initialize database
        db = DatabaseAdapter()
        
        # Load fresh data
        fresh_data = db.load_ohlc_data(dataset_name)
        if fresh_data is None or len(fresh_data) == 0:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found or empty")
        
        # Check if model is trained
        if model_type not in trained_models_storage:
            raise HTTPException(status_code=400, detail=f"{model_type} model not trained. Please train the model first.")
        
        model_info = trained_models_storage[model_type]
        model = model_info['model']
        
        # Generate features and predictions based on model type
        if model_type == 'volatility':
            return await generate_volatility_predictions(model, fresh_data, config)
        elif model_type == 'direction':
            return await generate_direction_predictions(model, fresh_data, config)
        elif model_type == 'profit_probability':
            return await generate_profit_probability_predictions(model, fresh_data, config)
        elif model_type == 'reversal':
            return await generate_reversal_predictions(model, fresh_data, config)
        else:
            raise HTTPException(status_code=400, detail=f"Model type {model_type} not supported")
            
    except Exception as e:
        logging.error(f"Prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction generation failed: {str(e)}")

async def generate_volatility_predictions(model, fresh_data, config):
    """Generate volatility predictions with comprehensive analysis - exact Streamlit functionality"""
    try:
        # Calculate features
        from features.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        features = ti.calculate_all_indicators(fresh_data)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=500, detail="Failed to calculate volatility features")
        
        # Make predictions
        predictions = model.predict(features)
        
        if predictions is None or len(predictions) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate volatility predictions")
        
        # Handle array length mismatch
        if len(predictions) != len(features):
            if len(predictions) < len(features):
                padded_predictions = np.full(len(features), np.nan)
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions
            else:
                predictions = predictions[:len(features)]
        
        # Create predictions dataframe
        pred_df = pd.DataFrame({
            'DateTime': features.index,
            'Predicted_Volatility': predictions
        })
        
        # Remove rows with NaN predictions
        pred_df = pred_df.dropna(subset=['Predicted_Volatility'])
        
        if len(pred_df) == 0:
            raise HTTPException(status_code=500, detail="No valid volatility predictions generated")
        
        # Calculate comprehensive statistics
        vol_data = pred_df['Predicted_Volatility']
        stats = {
            'mean': float(vol_data.mean()),
            'median': float(vol_data.median()),
            'std': float(vol_data.std()),
            'var': float(vol_data.var()),
            'skewness': float(vol_data.skew()),
            'kurtosis': float(vol_data.kurtosis()),
            'min': float(vol_data.min()),
            'max': float(vol_data.max()),
            'current_volatility': float(vol_data.iloc[-1])
        }
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = [float(np.percentile(vol_data, p)) for p in percentiles]
        
        # Rolling statistics
        vol_sample = vol_data.tail(500)
        rolling_mean = vol_sample.rolling(20).mean()
        rolling_std = vol_sample.rolling(20).std()
        
        # Autocorrelation
        lags = range(1, min(21, len(vol_sample)//4))
        autocorr = [float(vol_sample.autocorr(lag=lag)) if not pd.isna(vol_sample.autocorr(lag=lag)) else 0.0 for lag in lags]
        
        # Volatility clustering
        vol_changes = vol_sample.diff().abs()
        high_vol_threshold = vol_changes.quantile(0.8)
        clusters = []
        cluster_start = None
        
        for i, change in enumerate(vol_changes):
            if not pd.isna(change) and change > high_vol_threshold:
                if cluster_start is None:
                    cluster_start = i
            else:
                if cluster_start is not None:
                    clusters.append(i - cluster_start)
                    cluster_start = None
        
        clustering_stats = {
            'total_clusters': len(clusters),
            'avg_cluster_length': float(np.mean(clusters)) if clusters else 0.0,
            'max_cluster_length': int(max(clusters)) if clusters else 0,
            'clustering_percentage': float(sum(clusters)/len(vol_sample)*100) if clusters else 0.0
        }
        
        # Regime detection
        window = 50
        vol_data_regime = vol_data.tail(200)
        rolling_mean_regime = vol_data_regime.rolling(window).mean()
        rolling_std_regime = vol_data_regime.rolling(window).std()
        
        regimes = []
        for i in range(len(vol_data_regime)):
            if i < window:
                regimes.append('Insufficient Data')
            else:
                current_vol = vol_data_regime.iloc[i]
                mean_val = rolling_mean_regime.iloc[i]
                std_val = rolling_std_regime.iloc[i]
                
                if current_vol > mean_val + std_val:
                    regimes.append('High Volatility')
                elif current_vol < mean_val - std_val:
                    regimes.append('Low Volatility')
                else:
                    regimes.append('Normal Volatility')
        
        regime_counts = pd.Series(regimes).value_counts().to_dict()
        
        # Prepare chart data
        chart_data = []
        for i, row in pred_df.iterrows():
            chart_data.append({
                'datetime': row['DateTime'].isoformat(),
                'predicted_volatility': float(row['Predicted_Volatility']),
                'date': row['DateTime'].strftime('%Y-%m-%d'),
                'time': row['DateTime'].strftime('%H:%M:%S')
            })
        
        return {
            'success': True,
            'model_type': 'volatility',
            'total_predictions': len(pred_df),
            'predictions': chart_data,
            'statistics': stats,
            'percentiles': {str(p): v for p, v in zip(percentiles, percentile_values)},
            'rolling_stats': {
                'current_mean': float(rolling_mean.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0.0,
                'current_std': float(rolling_std.iloc[-1]) if not pd.isna(rolling_std.iloc[-1]) else 0.0,
            },
            'autocorrelation': {str(lag): corr for lag, corr in zip(lags, autocorr)},
            'clustering': clustering_stats,
            'regimes': regime_counts
        }
        
    except Exception as e:
        logging.error(f"Volatility prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Volatility prediction failed: {str(e)}")

async def generate_direction_predictions(model, fresh_data, config):
    """Generate direction predictions with comprehensive analysis"""
    try:
        # Calculate direction-specific features
        from features.direction_technical_indicators import DirectionTechnicalIndicators
        dti = DirectionTechnicalIndicators()
        features = dti.calculate_all_direction_indicators(fresh_data)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=500, detail="Failed to calculate direction features")
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        if predictions is None or len(predictions) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate direction predictions")
        
        # Ensure arrays are same length
        if len(predictions) != len(features):
            if len(predictions) < len(features):
                padded_predictions = np.full(len(features), np.nan)
                padded_predictions[:len(predictions)] = predictions
                predictions = padded_predictions
                
                if probabilities is not None:
                    padded_probs = np.full((len(features), probabilities.shape[1]), np.nan)
                    padded_probs[:len(probabilities)] = probabilities
                    probabilities = padded_probs
            else:
                predictions = predictions[:len(features)]
                if probabilities is not None:
                    probabilities = probabilities[:len(features)]
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'DateTime': features.index,
            'Direction': ['Bullish' if p == 1 else 'Bearish' for p in predictions],
            'Confidence': [float(np.max(prob)) if probabilities is not None and not np.isnan(prob).all() else 0.5 for prob in probabilities] if probabilities is not None else [0.5] * len(predictions)
        })
        
        # Remove NaN predictions
        pred_df = pred_df.dropna(subset=['DateTime'])
        
        # Calculate statistics
        bullish_count = len(pred_df[pred_df['Direction'] == 'Bullish'])
        bearish_count = len(pred_df[pred_df['Direction'] == 'Bearish'])
        avg_confidence = pred_df['Confidence'].mean()
        current_direction = pred_df['Direction'].iloc[-1]
        current_confidence = pred_df['Confidence'].iloc[-1]
        
        # Prepare chart data
        chart_data = []
        for i, row in pred_df.iterrows():
            chart_data.append({
                'datetime': row['DateTime'].isoformat(),
                'direction': row['Direction'],
                'confidence': float(row['Confidence']),
                'date': row['DateTime'].strftime('%Y-%m-%d'),
                'time': row['DateTime'].strftime('%H:%M:%S')
            })
        
        return {
            'success': True,
            'model_type': 'direction',
            'total_predictions': len(pred_df),
            'predictions': chart_data,
            'statistics': {
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'bullish_percentage': float(bullish_count / len(pred_df) * 100),
                'bearish_percentage': float(bearish_count / len(pred_df) * 100),
                'average_confidence': float(avg_confidence),
                'current_direction': current_direction,
                'current_confidence': float(current_confidence)
            }
        }
        
    except Exception as e:
        logging.error(f"Direction prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Direction prediction failed: {str(e)}")

async def generate_profit_probability_predictions(model, fresh_data, config):
    """Generate profit probability predictions"""
    try:
        # Calculate features
        from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
        pti = ProfitProbabilityTechnicalIndicators()
        features = pti.calculate_all_profit_probability_indicators(fresh_data)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=500, detail="Failed to calculate profit probability features")
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        if predictions is None or len(predictions) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate profit probability predictions")
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'DateTime': features.index,
            'Profit_Probability': ['High' if p == 1 else 'Low' for p in predictions],
            'Confidence': [float(np.max(prob)) if probabilities is not None and not np.isnan(prob).all() else 0.5 for prob in probabilities] if probabilities is not None else [0.5] * len(predictions)
        })
        
        # Remove NaN predictions
        pred_df = pred_df.dropna(subset=['DateTime'])
        
        # Calculate statistics
        high_profit_count = len(pred_df[pred_df['Profit_Probability'] == 'High'])
        low_profit_count = len(pred_df[pred_df['Profit_Probability'] == 'Low'])
        
        # Prepare chart data
        chart_data = []
        for i, row in pred_df.iterrows():
            chart_data.append({
                'datetime': row['DateTime'].isoformat(),
                'profit_probability': row['Profit_Probability'],
                'confidence': float(row['Confidence']),
                'date': row['DateTime'].strftime('%Y-%m-%d'),
                'time': row['DateTime'].strftime('%H:%M:%S')
            })
        
        return {
            'success': True,
            'model_type': 'profit_probability',
            'total_predictions': len(pred_df),
            'predictions': chart_data,
            'statistics': {
                'high_profit_count': high_profit_count,
                'low_profit_count': low_profit_count,
                'high_profit_percentage': float(high_profit_count / len(pred_df) * 100),
                'low_profit_percentage': float(low_profit_count / len(pred_df) * 100),
                'average_confidence': float(pred_df['Confidence'].mean()),
                'current_probability': pred_df['Profit_Probability'].iloc[-1],
                'current_confidence': float(pred_df['Confidence'].iloc[-1])
            }
        }
        
    except Exception as e:
        logging.error(f"Profit probability prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Profit probability prediction failed: {str(e)}")

async def generate_reversal_predictions(model, fresh_data, config):
    """Generate reversal predictions"""
    try:
        # Calculate features
        from features.reversal_technical_indicators import ReversalTechnicalIndicators
        rti = ReversalTechnicalIndicators()
        features = rti.calculate_all_reversal_indicators(fresh_data)
        
        if features is None or len(features) == 0:
            raise HTTPException(status_code=500, detail="Failed to calculate reversal features")
        
        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        if predictions is None or len(predictions) == 0:
            raise HTTPException(status_code=500, detail="Failed to generate reversal predictions")
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'DateTime': features.index,
            'Reversal_Signal': ['Bullish Reversal' if p == 1 else 'Bearish Reversal' if p == 2 else 'No Reversal' for p in predictions],
            'Confidence': [float(np.max(prob)) if probabilities is not None and not np.isnan(prob).all() else 0.5 for prob in probabilities] if probabilities is not None else [0.5] * len(predictions)
        })
        
        # Remove NaN predictions
        pred_df = pred_df.dropna(subset=['DateTime'])
        
        # Calculate statistics
        bullish_reversal_count = len(pred_df[pred_df['Reversal_Signal'] == 'Bullish Reversal'])
        bearish_reversal_count = len(pred_df[pred_df['Reversal_Signal'] == 'Bearish Reversal'])
        no_reversal_count = len(pred_df[pred_df['Reversal_Signal'] == 'No Reversal'])
        
        # Prepare chart data
        chart_data = []
        for i, row in pred_df.iterrows():
            chart_data.append({
                'datetime': row['DateTime'].isoformat(),
                'reversal_signal': row['Reversal_Signal'],
                'confidence': float(row['Confidence']),
                'date': row['DateTime'].strftime('%Y-%m-%d'),
                'time': row['DateTime'].strftime('%H:%M:%S')
            })
        
        return {
            'success': True,
            'model_type': 'reversal',
            'total_predictions': len(pred_df),
            'predictions': chart_data,
            'statistics': {
                'bullish_reversal_count': bullish_reversal_count,
                'bearish_reversal_count': bearish_reversal_count,
                'no_reversal_count': no_reversal_count,
                'bullish_reversal_percentage': float(bullish_reversal_count / len(pred_df) * 100),
                'bearish_reversal_percentage': float(bearish_reversal_count / len(pred_df) * 100),
                'no_reversal_percentage': float(no_reversal_count / len(pred_df) * 100),
                'average_confidence': float(pred_df['Confidence'].mean()),
                'current_signal': pred_df['Reversal_Signal'].iloc[-1],
                'current_confidence': float(pred_df['Confidence'].iloc[-1])
            }
        }
        
    except Exception as e:
        logging.error(f"Reversal prediction generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reversal prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)