from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any, List
import uvicorn
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Import your existing modules
try:
    from utils.database_adapter import DatabaseAdapter
except ImportError:
    from utils.postgres_database import PostgresTradingDatabase as DatabaseAdapter

try:
    from models.xgboost_models import QuantTradingModels
except ImportError:
    class QuantTradingModels:
        def __init__(self):
            self.models = {}
        def train_all_models(self, *args, **kwargs):
            return {}
        def predict(self, *args, **kwargs):
            return [], None

try:
    from features.technical_indicators import TechnicalIndicators
except ImportError:
    class TechnicalIndicators:
        @staticmethod
        def calculate_all_indicators(df):
            return df

try:
    from utils.realtime_data import IndianMarketData
except ImportError:
    class IndianMarketData:
        def fetch_realtime_data(self, *args, **kwargs):
            return None
        def is_market_open(self):
            return False

try:
    from utils.data_processing import DataProcessor
except ImportError:
    class DataProcessor:
        pass

import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for data persistence
try:
    trading_db = DatabaseAdapter()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"⚠️ Database initialization error: {e}")
    trading_db = None

model_trainer = None
current_data = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_trainer, current_data, trading_db
    try:
        if trading_db is None:
            print("⚠️ Database not available, initializing fallback")
            trading_db = type('MockDB', (), {
                'load_ohlc_data': lambda self, x: None,
                'save_model_results': lambda self, x, y: False,
                'save_trained_models': lambda self, x: False,
                'get_model_results': lambda self: [],
                'get_predictions': lambda self: [],
                'get_database_info': lambda self: {'total_datasets': 0, 'total_models': 0}
            })()
        
        model_trainer = QuantTradingModels()
        current_data = trading_db.load_ohlc_data("main_dataset")
        print("✅ API initialized successfully")
    except Exception as e:
        print(f"⚠️ API initialization warning: {e}")
        logger.error(traceback.format_exc())
    
    yield
    
    # Shutdown (if needed)
    pass

app = FastAPI(title="TribexAlpha API", version="1.0.0", lifespan=lifespan)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/data/summary")
async def get_data_summary():
    """Get current data summary"""
    global current_data, trading_db

    try:
        if trading_db is None:
            return {"error": "Database not available", "has_data": False}
            
        if current_data is None:
            current_data = trading_db.load_ohlc_data("main_dataset")

        if current_data is None:
            return {"error": "No data available", "has_data": False}

        # Simplified summary without DataProcessor dependency
        date_range = {}
        try:
            # Check if index is datetime-like
            if hasattr(current_data.index.min(), 'isoformat'):
                date_range = {
                    "start": current_data.index.min().isoformat(),
                    "end": current_data.index.max().isoformat(),
                    "days": (current_data.index.max() - current_data.index.min()).days
                }
            else:
                date_range = {
                    "start": str(current_data.index.min()),
                    "end": str(current_data.index.max()),
                    "days": current_data.index.max() - current_data.index.min()
                }
        except Exception:
            date_range = {
                "start": "N/A",
                "end": "N/A", 
                "days": 0
            }

        return {
            "has_data": True,
            "total_rows": len(current_data),
            "date_range": date_range,
            "price_summary": {
                "current_price": float(current_data['Close'].iloc[-1]),
                "price_change": float(current_data['Close'].iloc[-1] - current_data['Close'].iloc[-2]),
                "high": float(current_data['High'].max()),
                "low": float(current_data['Low'].min())
            },
            "returns": {
                "daily_mean": float(current_data['Close'].pct_change().mean()),
                "daily_std": float(current_data['Close'].pct_change().std())
            }
        }
    except Exception as e:
        logger.error(f"Error in get_data_summary: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "has_data": False}

@app.get("/api/data/chart")
async def get_chart_data(period: str = "30d"):
    """Get chart data for specified period"""
    global current_data

    if current_data is None:
        raise HTTPException(status_code=404, detail="No data available")

    # Filter data based on period
    if period == "30d":
        start_date = current_data.index.max() - timedelta(days=30)
    elif period == "90d":
        start_date = current_data.index.max() - timedelta(days=90)
    elif period == "1y":
        start_date = current_data.index.max() - timedelta(days=365)
    else:
        start_date = current_data.index.min()

    filtered_data = current_data[current_data.index >= start_date]

    chart_data = []
    for idx, row in filtered_data.iterrows():
        try:
            date_str = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
        except:
            date_str = str(idx)
        
        chart_data.append({
            "date": date_str,
            "open": float(row["Open"]) if pd.notna(row["Open"]) else 0,
            "high": float(row["High"]) if pd.notna(row["High"]) else 0,
            "low": float(row["Low"]) if pd.notna(row["Low"]) else 0,
            "close": float(row["Close"]) if pd.notna(row["Close"]) else 0,
            "volume": float(row.get("Volume", 0)) if pd.notna(row.get("Volume", 0)) else 0
        })

    return {"data": chart_data}

@app.get("/api/models/status")
async def get_models_status():
    """Get status of trained models"""
    global model_trainer

    try:
        if model_trainer is None:
            return {"models": {}, "total_models": 0}

        models_info = {}
        if hasattr(model_trainer, 'models') and model_trainer.models:
            for name, model_data in model_trainer.models.items():
                if model_data:
                    models_info[name] = {
                        "name": name.replace('_', ' ').title(),
                        "trained": True,
                        "task_type": model_data.get('task_type', 'unknown'),
                        "trained_at": model_data.get('trained_at', 'unknown')
                    }

        return {
            "models": models_info,
            "total_models": len(models_info)
        }
    except Exception as e:
        logger.error(f"Error in get_models_status: {str(e)}")
        logger.error(traceback.format_exc())
        return {"models": {}, "total_models": 0, "error": str(e)}

@app.post("/api/models/train")
async def train_models(models_to_train: List[str]):
    """Train selected models"""
    global model_trainer, current_data

    if current_data is None:
        raise HTTPException(status_code=400, detail="No data available for training")

    if model_trainer is None:
        model_trainer = QuantTradingModels()

    try:
        # Calculate features if needed
        features_data = TechnicalIndicators.calculate_all_indicators(current_data)
        features_data = features_data.dropna()

        # Train models
        results = model_trainer.train_all_models(features_data, 0.8)

        # Save to database
        for model_name, model_result in results.items():
            if model_result is not None:
                model_data = {
                    'metrics': model_result['metrics'],
                    'task_type': model_result['task_type'],
                    'trained_at': datetime.now().isoformat()
                }
                trading_db.save_model_results(model_name, model_data)

        trading_db.save_trained_models(model_trainer.models)

        return {"status": "success", "trained_models": list(results.keys())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/predictions/{model_name}")
async def get_predictions(model_name: str, period: str = "30d"):
    """Get predictions for a specific model"""
    global model_trainer, current_data

    if model_trainer is None or current_data is None:
        raise HTTPException(status_code=404, detail="Models or data not available")

    try:
        # Prepare features
        features_data = TechnicalIndicators.calculate_all_indicators(current_data)
        features_data = features_data.dropna()

        # Filter by period
        if period == "30d":
            start_date = features_data.index.max() - timedelta(days=30)
        elif period == "90d":
            start_date = features_data.index.max() - timedelta(days=90)
        else:
            start_date = features_data.index.min()

        features_filtered = features_data[features_data.index >= start_date]
        price_filtered = current_data[current_data.index >= start_date]

        # Generate predictions
        predictions, probabilities = model_trainer.predict(model_name, features_filtered)

        # Format response
        result = []
        for i, (idx, price_row) in enumerate(price_filtered.iterrows()):
            try:
                date_str = idx.isoformat() if hasattr(idx, 'isoformat') else str(idx)
            except:
                date_str = str(idx)
                
            pred_data = {
                "date": date_str,
                "price": float(price_row["Close"]) if pd.notna(price_row["Close"]) else 0,
                "prediction": int(predictions[i]) if i < len(predictions) else None
            }

            if probabilities is not None and i < len(probabilities):
                if probabilities.ndim > 1:
                    pred_data["confidence"] = float(np.max(probabilities[i]))
                else:
                    pred_data["confidence"] = float(probabilities[i])

            result.append(pred_data)

        return {
            "model_name": model_name,
            "predictions": result,
            "total_predictions": len(result)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/realtime/nifty")
async def get_realtime_nifty():
    """Get real-time NIFTY 50 data"""
    try:
        market_data = IndianMarketData()
        df = market_data.fetch_realtime_data("^NSEI", period="1d", interval="5m")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No real-time data available")

        # Get current price info
        current_price = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close * 100) if previous_close != 0 else 0

        # Calculate technical indicators
        tech_df = TechnicalIndicators.calculate_all_indicators(df)

        return {
            "symbol": "NIFTY 50",
            "current_price": float(current_price),
            "price_change": float(price_change),
            "price_change_pct": float(price_change_pct),
            "high": float(df['High'].iloc[-1]),
            "low": float(df['Low'].iloc[-1]),
            "volume": float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0,
            "market_open": market_data.is_market_open(),
            "last_updated": datetime.now().isoformat(),
            "technical_indicators": {
                "rsi": float(tech_df['rsi'].iloc[-1]) if 'rsi' in tech_df.columns and not pd.isna(tech_df['rsi'].iloc[-1]) else None,
                "macd": float(tech_df['macd_histogram'].iloc[-1]) if 'macd_histogram' in tech_df.columns and not pd.isna(tech_df['macd_histogram'].iloc[-1]) else None,
                "bb_position": float(tech_df['bb_position'].iloc[-1]) if 'bb_position' in tech_df.columns and not pd.isna(tech_df['bb_position'].iloc[-1]) else None
            },
            "chart_data": [
                {
                    "date": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    "open": float(row["Open"]) if pd.notna(row["Open"]) else 0,
                    "high": float(row["High"]) if pd.notna(row["High"]) else 0,
                    "low": float(row["Low"]) if pd.notna(row["Low"]) else 0,
                    "close": float(row["Close"]) if pd.notna(row["Close"]) else 0,
                    "volume": float(row.get("Volume", 0)) if pd.notna(row.get("Volume", 0)) else 0
                }
                for idx, row in df.tail(50).iterrows()
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Real-time data error: {str(e)}")

@app.post("/api/data/upload")
async def upload_data(file_content: str, filename: str):
    """Handle file upload and processing"""
    global current_data

    try:
        # This would handle file processing - simplified for now
        # In a real implementation, you'd process the uploaded CSV content
        return {"status": "success", "message": "File upload endpoint ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Serve React static files (when built)
try:
    app.mount("/static", StaticFiles(directory="build/static"), name="static")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all routes"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        return FileResponse("build/index.html")
except Exception:
    # Build directory doesn't exist yet - React dev server will handle serving
    pass

@app.get('/api/database/info')
async def get_database_info():
    """Get database information including datasets, models, and predictions."""
    global trading_db
    try:
        if trading_db is None:
            return {"error": "Database not available", "total_datasets": 0, "total_models": 0}
            
        info = trading_db.get_database_info()

        # Get model results
        model_results = []
        try:
            model_results_data = trading_db.get_model_results()
            for result in model_results_data:
                model_results.append({
                    'name': result['model_name'],
                    'results': result['results_json']
                })
        except Exception as e:
            print(f"Error fetching model results: {e}")

        # Get predictions
        predictions = []
        try:
            predictions_data = trading_db.get_predictions()
            for result in predictions_data:
                pred_data = result['predictions_json']
                predictions.append({
                    'model_name': result['model_name'],
                    'shape': f"({len(pred_data)}, {len(pred_data[0].keys()) if pred_data else 0})",
                    'columns': ', '.join(pred_data[0].keys()) if pred_data else '',
                    'created_at': result['created_at']
                })
        except Exception as e:
            print(f"Error fetching predictions: {e}")

        info['model_results'] = model_results
        info['predictions'] = predictions

        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/database/dataset/{dataset_name}')
async def delete_dataset(dataset_name: str):
    """Delete a dataset from the database."""
    try:
        success = trading_db.delete_dataset(dataset_name)
        return {'success': success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/database/export/{dataset_name}')
async def export_dataset(dataset_name: str):
    """Export a dataset as CSV."""
    try:
        data = trading_db.load_ohlc_data(dataset_name)

        if data is not None:
            csv_data = data.to_csv()
            return FileResponse(csv_data, media_type="text/csv", filename=f"{dataset_name}.csv")
        else:
            raise HTTPException(status_code=404, detail='Dataset not found')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete('/api/database/clear-all')
async def clear_all_database():
    """Clear all data from the database."""
    try:
        success = trading_db.clear_all_data()
        return {'success': success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/data/load')
async def load_dataset(dataset_name: str):
    """Load a dataset into session."""
    try:
        loaded_data = trading_db.load_ohlc_data(dataset_name)

        if loaded_data is not None:
            # Store in session or global variable (simplified for this example)
            return {'success': True, 'message': f'Loaded {len(loaded_data)} rows'}
        else:
            return {'success': False, 'error': 'Failed to load dataset'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)