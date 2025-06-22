#!/usr/bin/env python3
"""
Quick model retraining script to restore trained models after app restart
"""

import pandas as pd
from utils.database import TradingDatabase
from models.xgboost_models import QuantTradingModels
from features.technical_indicators import TechnicalIndicators

def retrain_all_models():
    """Retrain all models and save to database"""
    print("Starting model retraining process...")
    
    # Load data from database
    db = DatabaseAdapter()
    data = db.load_ohlc_data("main_dataset")
    
    if data is None:
        print("ERROR: No data found in database!")
        return False
    
    print(f"Loaded dataset with {len(data)} rows")
    
    # Calculate features
    print("Calculating technical indicators...")
    features_data = TechnicalIndicators.calculate_all_indicators(data)
    
    # Initialize model trainer
    print("Initializing model trainer...")
    model_trainer = QuantTradingModels()
    
    # Train all models
    print("Training all models...")
    training_results = model_trainer.train_all_models(features_data)
    
    # Save models to database
    print("Saving trained models to database...")
    success = db.save_trained_models(model_trainer.models)
    
    if success:
        print("✅ All models successfully trained and saved!")
        
        # Print summary
        for model_name, result in training_results.items():
            if result and 'accuracy' in result:
                accuracy = result['accuracy']
                print(f"  {model_name}: {accuracy:.3f} accuracy")
        
        return True
    else:
        print("❌ Failed to save models to database")
        return False

if __name__ == "__main__":
    success = retrain_all_models()
    exit(0 if success else 1)