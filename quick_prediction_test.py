#!/usr/bin/env python3
"""Quick test to verify prediction functionality"""
import sys
sys.path.append('.')

def test_prediction_system():
    try:
        from models.xgboost_models import QuantTradingModels
        from utils.database_adapter import get_trading_database
        
        # Load data and models
        db = get_trading_database()
        data = db.load_ohlc_data('main_dataset')
        
        # Initialize trainer
        trainer = QuantTradingModels()
        
        # Prepare features
        features = trainer.prepare_features(data.tail(20))
        
        # Test with available models
        for model_name in ['direction', 'trend_sideways', 'magnitude']:
            if model_name in trainer.models:
                predictions, probabilities = trainer.predict(model_name, features.tail(5))
                print(f"{model_name}: {len(predictions)} predictions generated successfully")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_prediction_system()
    print(f"Prediction system working: {success}")