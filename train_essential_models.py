#!/usr/bin/env python3
"""
Essential model training script for immediate predictions
"""
import pandas as pd
import numpy as np
from utils.database import TradingDatabase
from models.xgboost_models import QuantTradingModels
from features.technical_indicators import TechnicalIndicators
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def train_essential_models():
    """Train essential models quickly for immediate use"""
    print("Starting essential model training...")
    
    try:
        # Load database
        db = TradingDatabase()
        
        # Load data
        print("Loading data from database...")
        data = db.load_ohlc_data("main_dataset")
        
        if data is None:
            print("No data found in database")
            return False
        
        # Use last 5000 rows for quick training
        recent_data = data.tail(5000).copy()
        print(f"Using {len(recent_data)} recent rows for training")
        
        # Calculate features
        print("Calculating technical indicators...")
        features_data = TechnicalIndicators.calculate_all_indicators(recent_data)
        
        # Remove any rows with NaN values
        features_data = features_data.dropna()
        print(f"After cleaning: {len(features_data)} rows available")
        
        if len(features_data) < 100:
            print("Not enough clean data for training")
            return False
        
        # Initialize model trainer
        model_trainer = QuantTradingModels()
        
        # Prepare features and targets
        X = model_trainer.prepare_features(features_data)
        targets = model_trainer.create_targets(features_data)
        
        # Train only essential models
        essential_models = ['direction', 'magnitude', 'trading_signal']
        results = {}
        
        for model_name in essential_models:
            if model_name in targets:
                print(f"Training {model_name} model...")
                
                y = targets[model_name]
                task_type = 'classification' if model_name in ['direction', 'trading_signal'] else 'regression'
                
                # Train the model
                result = model_trainer.train_model(model_name, X, y, task_type)
                
                if result and 'accuracy' in result:
                    results[model_name] = result
                    print(f"✓ {model_name}: {result['accuracy']:.3f} accuracy")
                    
                    # Save individual model results
                    model_data = {
                        'metrics': result,
                        'task_type': task_type,
                        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'accuracy': result['accuracy']
                    }
                    db.save_model_results(model_name, model_data)
                else:
                    print(f"✗ Failed to train {model_name}")
        
        # Save trained model objects
        if model_trainer.models:
            print("Saving trained models to database...")
            success = db.save_trained_models(model_trainer.models)
            if success:
                print("✓ Models saved successfully")
                return True
            else:
                print("✗ Failed to save models")
                return False
        else:
            print("No models were trained successfully")
            return False
            
    except Exception as e:
        print(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    success = train_essential_models()
    if success:
        print("\n✓ Essential models trained and ready for predictions!")
    else:
        print("\n✗ Training failed")
    exit(0 if success else 1)