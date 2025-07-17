#!/usr/bin/env python3

"""Script to force-start the prediction pipeline if models are available"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_prediction_pipeline import LivePredictionPipeline
from models.model_manager import ModelManager
import time

def main():
    print("=== Force Starting Prediction Pipeline ===")
    
    # Check if we have models
    model_manager = ModelManager()
    available_models = [name for name in ['direction', 'volatility', 'profit_probability', 'reversal'] 
                       if model_manager.is_model_trained(name)]
    
    print(f"Available models: {available_models}")
    
    if not available_models:
        print("❌ No models available. Cannot start prediction pipeline.")
        return
    
    # Create a pipeline with dummy credentials (for testing the model loading logic)
    pipeline = LivePredictionPipeline("dummy_token", "dummy_key")
    
    # Check if models are loaded in the pipeline
    pipeline_models = [name for name in ['direction', 'volatility', 'profit_probability', 'reversal'] 
                      if pipeline.model_manager.is_model_trained(name)]
    
    print(f"Pipeline models: {pipeline_models}")
    
    if pipeline_models:
        print("✅ Models are available in the pipeline. The issue is with the live connection.")
        print("📌 SOLUTION: Go to Live Data page and:")
        print("   1. Disconnect if already connected")
        print("   2. Enter your Upstox credentials")
        print("   3. Click 'Connect' to start the prediction pipeline")
        print("   4. Make sure 'Nifty 50' is selected in instruments")
        print("   5. The system will automatically start generating predictions")
    else:
        print("❌ Models not loading in pipeline. There may be a session state issue.")

if __name__ == "__main__":
    main()