#!/usr/bin/env python3

"""Test script to manually start the prediction pipeline"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_prediction_pipeline import LivePredictionPipeline
from models.model_manager import ModelManager
import time

def test_model_loading():
    """Test model loading directly"""
    print("🔍 Testing model loading directly...")
    
    # Create model manager
    model_manager = ModelManager()
    
    # Check trained models
    print(f"Trained models: {list(model_manager.trained_models.keys())}")
    
    # Test each model
    for model_name in ['direction', 'volatility', 'profit_probability', 'reversal']:
        is_trained = model_manager.is_model_trained(model_name)
        print(f"{model_name}: {'✅' if is_trained else '❌'}")
        
        if is_trained:
            model_data = model_manager.trained_models.get(model_name, {})
            has_model = 'model' in model_data or 'ensemble' in model_data
            has_scaler = 'scaler' in model_data
            has_features = 'feature_names' in model_data
            print(f"  - Model: {has_model}, Scaler: {has_scaler}, Features: {has_features}")
    
    return len(model_manager.trained_models) > 0

def test_pipeline_start():
    """Test starting the prediction pipeline"""
    print("🚀 Testing prediction pipeline start...")
    
    # Test credentials (these would normally come from user input)
    access_token = "test_token"
    api_key = "test_key"
    
    # Create pipeline
    pipeline = LivePredictionPipeline(access_token, api_key)
    
    # Try to start (will fail due to auth, but we want to see model loading)
    try:
        result = pipeline.start_pipeline()
        print(f"Pipeline start result: {result}")
    except Exception as e:
        print(f"Expected error (auth): {e}")
    
    return True

if __name__ == "__main__":
    print("=== Testing Prediction Pipeline ===")
    
    # Test model loading
    models_loaded = test_model_loading()
    print(f"\nModels loaded: {models_loaded}")
    
    # Test pipeline start
    pipeline_tested = test_pipeline_start()
    print(f"Pipeline tested: {pipeline_tested}")