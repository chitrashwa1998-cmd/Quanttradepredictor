#!/usr/bin/env python3

"""Test script to verify the prediction pipeline can access new candles"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_prediction_pipeline import LivePredictionPipeline
import pandas as pd

def test_prediction_pipeline():
    """Test if the prediction pipeline can detect new candles"""
    print("=== Testing Prediction Pipeline ===")
    
    # Create dummy pipeline to test candle detection
    pipeline = LivePredictionPipeline("dummy_token", "dummy_key")
    
    # Test candle detection logic
    instrument_key = "NSE_INDEX|Nifty 50"
    
    # Get latest OHLC data
    try:
        ohlc_data = pipeline.live_data_manager.get_live_ohlc(instrument_key, rows=10)
        
        if ohlc_data is not None and len(ohlc_data) > 0:
            print(f"✅ Successfully retrieved OHLC data: {len(ohlc_data)} rows")
            print(f"📊 Latest candle timestamp: {ohlc_data.index[-1]}")
            print(f"📊 Latest candle close price: {ohlc_data['Close'].iloc[-1]:.2f}")
            
            # Test candle progression
            if len(ohlc_data) >= 3:
                last_3_candles = ohlc_data.tail(3)
                print(f"📈 Last 3 candles:")
                for i, (timestamp, row) in enumerate(last_3_candles.iterrows()):
                    print(f"   {i+1}. {timestamp} - Close: {row['Close']:.2f}")
                    
                # Check if candles are progressing in time
                timestamps = last_3_candles.index.tolist()
                if timestamps == sorted(timestamps):
                    print("✅ Candles are properly ordered in time")
                    
                    # Check 5-minute intervals
                    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
                    expected_diff = 5 * 60  # 5 minutes in seconds
                    
                    if all(abs(diff - expected_diff) < 30 for diff in time_diffs):  # Allow 30s tolerance
                        print("✅ Candles follow proper 5-minute intervals")
                        return True
                    else:
                        print(f"⚠️ Candle intervals: {time_diffs} (expected ~{expected_diff}s)")
                        return False
                else:
                    print("❌ Candles are not properly ordered")
                    return False
            else:
                print("⚠️ Need at least 3 candles to test progression")
                return False
        else:
            print("❌ No OHLC data available")
            return False
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction_pipeline()
    
    if success:
        print("\n🎯 PREDICTION PIPELINE READY!")
        print("✅ New candles are being created properly")
        print("✅ Prediction pipeline can access the data")
        print("✅ Next prediction should trigger when current candle completes")
    else:
        print("\n❌ PREDICTION PIPELINE NEEDS ATTENTION")
        print("Please check the candle creation logic")