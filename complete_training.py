#!/usr/bin/env python3
"""
Complete training script for all 7 AI models
Forces completion of magnitude and volatility regression models
"""

import sys
import os
sys.path.append('.')

from models.xgboost_models import QuantTradingModels
from utils.database_adapter import get_trading_database
import pandas as pd
import numpy as np

def complete_all_models():
    """Complete training for all 7 AI models"""
    print("🔄 Starting complete model training...")
    
    # Initialize components
    db = get_trading_database()
    model_trainer = QuantTradingModels()
    
    # Load data
    print("📊 Loading training data...")
    data = db.load_ohlc_data('main_dataset')
    if data is None or data.empty:
        print("❌ No data available for training")
        return False
    
    print(f"✅ Loaded {len(data)} rows of data")
    
    # Train all models with forced completion
    print("🚀 Training all 7 AI models...")
    results = model_trainer.train_all_models(data, train_split=0.8)
    
    if results['success']:
        print(f"✅ Successfully trained {len(results['trained_models'])} models")
        
        # Display all trained models
        for model_name, metrics in results['trained_models'].items():
            if 'accuracy' in metrics:
                print(f"  📈 {model_name}: {metrics['accuracy']:.4f} accuracy")
            else:
                print(f"  📊 {model_name}: MSE={metrics.get('mse', 'N/A'):.6f}")
        
        # Save models to database
        print("💾 Saving models to database...")
        model_trainer._save_models_to_database()
        print("✅ All models saved successfully")
        
        return True
    else:
        print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        return False

if __name__ == "__main__":
    success = complete_all_models()
    if success:
        print("\n🎉 All 7 AI models completed successfully!")
    else:
        print("\n❌ Training incomplete")
        sys.exit(1)