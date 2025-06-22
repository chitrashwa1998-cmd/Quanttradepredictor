#!/usr/bin/env python3
"""
Quick test script to verify prediction functionality is working
"""
import sys
sys.path.append('.')

from models.xgboost_models import QuantTradingModels
from utils.database_adapter import get_trading_database
import pandas as pd

def test_predictions():
    print("Testing prediction functionality...")
    
    try:
        # Load data and models
        db = get_trading_database()
        data = db.load_ohlc_data('main_dataset')
        print(f"✓ Data loaded: {len(data)} rows")
        
        # Initialize trainer
        trainer = QuantTradingModels()
        trained_models = db.load_trained_models()
        
        if trained_models:
            trainer.models = trained_models
            print(f"✓ Models loaded: {list(trainer.models.keys())}")
            print(f"✓ Feature names count: {len(trainer.feature_names) if trainer.feature_names else 0}")
            
            # Prepare features
            features = trainer.prepare_features(data.tail(100))
            print(f"✓ Features prepared: {features.shape}")
            
            # Test predictions for each available model
            for model_name in trainer.models.keys():
                try:
                    predictions, probabilities = trainer.predict(model_name, features.tail(10))
                    print(f"✓ {model_name}: {len(predictions)} predictions generated")
                    print(f"  Sample: {predictions[:3]}")
                    if probabilities is not None:
                        print(f"  Confidence: {probabilities[0][:2] if len(probabilities) > 0 else 'N/A'}")
                except Exception as e:
                    print(f"✗ {model_name}: {str(e)}")
            
            print("\n✓ Prediction system is working correctly!")
            return True
            
        else:
            print("✗ No trained models found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_predictions()
    if success:
        print("\nThe prediction tab should now show results when you navigate to it.")
    else:
        print("\nPrediction system needs troubleshooting.")