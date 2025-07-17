#!/usr/bin/env python3

"""Debug script to check pipeline status and manually start if needed"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_prediction_pipeline import LivePredictionPipeline
from models.model_manager import ModelManager
from utils.database_adapter import get_trading_database
import time

def check_database_models():
    """Check what models are in the database"""
    print("🔍 Checking database models...")
    
    db = get_trading_database()
    loaded_models = db.load_trained_models()
    
    if loaded_models:
        print(f"✅ Found {len(loaded_models)} models in database: {list(loaded_models.keys())}")
        for model_name, model_data in loaded_models.items():
            has_model = 'model' in model_data or 'ensemble' in model_data
            has_scaler = 'scaler' in model_data
            has_features = 'feature_names' in model_data
            print(f"  {model_name}: Model={has_model}, Scaler={has_scaler}, Features={has_features}")
    else:
        print("❌ No models found in database")
    
    return loaded_models

def check_model_manager():
    """Check if ModelManager can load models"""
    print("🔍 Checking ModelManager...")
    
    model_manager = ModelManager()
    print(f"✅ ModelManager has {len(model_manager.trained_models)} trained models: {list(model_manager.trained_models.keys())}")
    
    for model_name in ['direction', 'volatility', 'profit_probability', 'reversal']:
        is_trained = model_manager.is_model_trained(model_name)
        print(f"  {model_name}: {'✅' if is_trained else '❌'}")
    
    return model_manager

def main():
    print("=== DEBUG: Pipeline Status ===")
    
    # Check database models
    db_models = check_database_models()
    
    # Check model manager
    model_manager = check_model_manager()
    
    # Show current state
    print("\n=== CURRENT STATUS ===")
    print(f"Database models: {len(db_models) if db_models else 0}")
    print(f"ModelManager models: {len(model_manager.trained_models)}")
    
    # Check if we have at least one model
    if len(model_manager.trained_models) > 0:
        print("✅ Models are available - pipeline should be able to start")
    else:
        print("❌ No models available - pipeline cannot start")
        
    print("\n=== RECOMMENDATION ===")
    if len(model_manager.trained_models) > 0:
        print("Models are loaded. Try connecting to Live Data with valid Upstox credentials.")
        print("The prediction pipeline should start automatically once connected.")
    else:
        print("No models found. Please go to Model Training page and train at least one model.")

if __name__ == "__main__":
    main()