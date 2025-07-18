#!/usr/bin/env python3
"""
Quick script to retrain the direction model with improved datetime filtering
"""
import pandas as pd
import numpy as np
from utils.database_adapter import get_trading_database

def retrain_direction_model():
    """Retrain the direction model with corrected filtering logic"""
    
    print("🔄 Retraining direction model with improved datetime filtering...")
    
    # Load database and data
    db = get_trading_database()
    data = db.load_ohlc_data("training_dataset")
    
    if data is None or len(data) == 0:
        print("❌ No training data available")
        return False
    
    print(f"✅ Loaded training data: {len(data)} rows")
    
    # Import direction model
    from models.direction_model import DirectionModel
    direction_model = DirectionModel()
    
    # Prepare features and target
    print("🔧 Preparing features...")
    features = direction_model.prepare_features(data)
    print(f"✅ Features prepared: {features.shape}")
    
    print("🎯 Creating target...")
    target = direction_model.create_target(data)
    print(f"✅ Target created: {len(target)} samples")
    
    # Train the model
    print("🚀 Training direction model...")
    training_result = direction_model.train(features, target)
    
    if training_result is None:
        print("❌ Training failed")
        return False
    
    print("✅ Direction model trained successfully!")
    
    # Save to database
    print("💾 Saving to database...")
    feature_names = getattr(direction_model, 'feature_names', [])
    print(f"Features to save: {len(feature_names)} - {feature_names[:5] if feature_names else 'None'}")
    
    models_to_save = {
        'direction': {
            'ensemble': direction_model.model,
            'scaler': direction_model.scaler,
            'feature_names': feature_names,
            'task_type': 'classification',
            'metrics': training_result.get('metrics', {}),
            'feature_importance': training_result.get('feature_importance', {})
        }
    }
    
    success = db.save_trained_models(models_to_save)
    if success:
        print("✅ Direction model saved to database successfully!")
        return True
    else:
        print("❌ Failed to save direction model to database")
        return False

if __name__ == "__main__":
    retrain_direction_model()