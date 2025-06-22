#!/usr/bin/env python3
"""
Quick training script using optimized data loading
"""
import pandas as pd
import numpy as np
from utils.database import TradingDatabase
from models.xgboost_models import QuantTradingModels
from features.technical_indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

def quick_train():
    print("Quick training process starting...")
    
    db = TradingDatabase()
    
    try:
        # Load the full dataset using the existing method
        print("Loading dataset using database method...")
        data = db.load_ohlc_data("main_dataset")
        
        if data is not None:
            # Use only the most recent 5000 rows to speed up training
            recent_data = data.tail(5000).copy()
            print(f"Using most recent {len(recent_data)} rows for quick training")
            
            # Calculate features
            print("Calculating technical indicators...")
            features_data = TechnicalIndicators.calculate_all_indicators(recent_data)
            
            # Initialize model trainer
            model_trainer = QuantTradingModels()
            
            # Prepare features
            X = model_trainer.prepare_features(features_data)
            targets = model_trainer.create_targets(features_data)
            
            print("Training essential models...")
            
            # Train direction model (classification)
            if 'direction' in targets:
                print("Training direction model...")
                y_dir = targets['direction']
                result = model_trainer.train_model('direction', X, y_dir, 'classification')
                if result:
                    db.save_model_results('direction', result)
                    print("✓ Direction model trained")
            
            # Train magnitude model (regression)
            if 'magnitude' in targets:
                print("Training magnitude model...")
                y_mag = targets['magnitude']
                result = model_trainer.train_model('magnitude', X, y_mag, 'regression')
                if result:
                    db.save_model_results('magnitude', result)
                    print("✓ Magnitude model trained")
            
            # Train trading signal model (classification)
            if 'trading_signal' in targets:
                print("Training trading signal model...")
                y_sig = targets['trading_signal']
                result = model_trainer.train_model('trading_signal', X, y_sig, 'classification')
                if result:
                    db.save_model_results('trading_signal', result)
                    print("✓ Trading signal model trained")
            
            # Save trained models
            if model_trainer.models:
                success = db.save_trained_models(model_trainer.models)
                if success:
                    print("✓ All models saved to database")
                    return True
                else:
                    print("✗ Failed to save models")
                    return False
            else:
                print("No models were trained")
                return False
        else:
            print("No data found in database")
            return False
    except Exception as e:
        print(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    success = quick_train()
    exit(0 if success else 1)