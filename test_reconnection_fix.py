#!/usr/bin/env python3

"""Test script to verify the reconnection fix works"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_prediction_pipeline import LivePredictionPipeline
from models.model_manager import ModelManager
import time

def test_reconnection_scenario():
    """Test the disconnect/reconnect scenario"""
    print("=== Testing Reconnection Scenario ===")
    
    # Step 1: Create first pipeline (simulating first connection)
    print("1. Creating first pipeline...")
    pipeline1 = LivePredictionPipeline("dummy_token", "dummy_key")
    
    # Add some dummy candle timestamps (simulating processed candles)
    pipeline1.last_candle_timestamps = {
        "NSE_INDEX|Nifty 50": "2025-07-17 12:00:00"
    }
    print(f"   Pipeline1 candle timestamps: {pipeline1.last_candle_timestamps}")
    
    # Step 2: Stop pipeline (simulating disconnect)
    print("2. Stopping pipeline...")
    pipeline1.stop_pipeline()
    print(f"   Pipeline1 candle timestamps after stop: {pipeline1.last_candle_timestamps}")
    
    # Step 3: Create new pipeline (simulating reconnection)
    print("3. Creating new pipeline (reconnection)...")
    pipeline2 = LivePredictionPipeline("dummy_token", "dummy_key")
    print(f"   Pipeline2 candle timestamps: {pipeline2.last_candle_timestamps}")
    
    # Verify the fix
    if not pipeline2.last_candle_timestamps:
        print("✅ SUCCESS: New pipeline has empty candle timestamps - predictions can restart")
        return True
    else:
        print("❌ FAILED: New pipeline still has candle timestamps - predictions may be blocked")
        return False

def test_model_loading():
    """Test that models are still loaded correctly"""
    print("\n=== Testing Model Loading ===")
    
    model_manager = ModelManager()
    available_models = [name for name in ['direction', 'volatility', 'profit_probability', 'reversal'] 
                       if model_manager.is_model_trained(name)]
    
    print(f"Available models: {available_models}")
    
    if available_models:
        print("✅ SUCCESS: Models are available for predictions")
        return True
    else:
        print("❌ FAILED: No models available")
        return False

if __name__ == "__main__":
    reconnection_test = test_reconnection_scenario()
    model_test = test_model_loading()
    
    print("\n=== FINAL RESULT ===")
    if reconnection_test and model_test:
        print("✅ All tests passed! The reconnection fix should work.")
        print("📌 Instructions for user:")
        print("   1. Go to Live Data page")
        print("   2. Click 'Disconnect' if connected")
        print("   3. Enter Upstox credentials")
        print("   4. Click 'Connect'")
        print("   5. Select 'Nifty 50' and wait for predictions")
    else:
        print("❌ Some tests failed. Please check the implementation.")