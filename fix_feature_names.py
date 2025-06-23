
#!/usr/bin/env python3
"""
Fix feature names issue by retraining models with proper feature persistence
"""
import pandas as pd
import numpy as np
from utils.database_adapter import get_trading_database
from models.xgboost_models import QuantTradingModels
from features.technical_indicators import TechnicalIndicators
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fix_feature_names():
    """Fix feature names issue by retraining essential models"""
    print("Fixing feature names issue...")
    
    try:
        # Load database
        db = get_trading_database()
        
        # Load data
        print("Loading data from database...")
        data = db.load_ohlc_data("main_dataset")
        
        if data is None:
            print("No data found in database")
            return False
        
        # Use last 10000 rows for quick training
        recent_data = data.tail(10000).copy()
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
        
        # Prepare features and targets - this will set feature_names properly
        print("Preparing features...")
        X = model_trainer.prepare_features(features_data)
        print(f"Feature names properly set: {len(model_trainer.feature_names)} features")
        
        targets = model_trainer.create_targets(features_data)
        
        # Train only direction model first to test
        model_name = 'direction'
        print(f"Training {model_name} model...")
        
        if model_name in targets:
            y = targets[model_name]
            task_type = 'classification'
            
            # Train the model
            result = model_trainer.train_model(model_name, X, y, task_type)
            
            if result and 'accuracy' in result:
                print(f"✓ {model_name}: {result['accuracy']:.3f} accuracy")
                
                # Test prediction to verify feature names work
                print("Testing prediction...")
                test_predictions, test_probabilities = model_trainer.predict(model_name, X.tail(10))
                print(f"✓ Prediction successful: {len(test_predictions)} predictions")
                
                # Save model results
                model_data = {
                    'metrics': result,
                    'task_type': task_type,
                    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'accuracy': result['accuracy']
                }
                db.save_model_results(model_name, model_data)
                
                # Save trained model objects
                print("Saving trained models to database...")
                success = db.save_trained_models(model_trainer.models)
                if success:
                    print("✓ Models saved successfully with feature names")
                    return True
                else:
                    print("✗ Failed to save models")
                    return False
            else:
                print(f"✗ Failed to train {model_name}")
                return False
        else:
            print(f"✗ Target {model_name} not found")
            return False
            
    except Exception as e:
        print(f"Error during fix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_feature_names()
    if success:
        print("\n✓ Feature names issue fixed! Models ready for predictions.")
    else:
        print("\n✗ Fix failed")
    exit(0 if success else 1)
