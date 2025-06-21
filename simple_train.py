#!/usr/bin/env python3
"""
Simple model training to get the dashboard functional quickly
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append('.')

def quick_train():
    """Train essential models quickly"""
    try:
        # Import modules
        from utils.database_adapter import get_trading_database
        from models.xgboost_models import QuantTradingModels
        
        print("Loading data...")
        db = get_trading_database()
        data = db.load_ohlc_data("main_dataset")
        
        if data is None or data.empty:
            print("No data found")
            return False
        
        print(f"Training on {len(data)} data points...")
        
        # Initialize models
        models = QuantTradingModels()
        
        # Quick training with smaller dataset for speed
        sample_size = min(500, len(data))  # Use last 500 points for quick training
        train_data = data.tail(sample_size).copy()
        
        print(f"Using {len(train_data)} points for quick training...")
        
        # Train just the essential models
        results = models.train_all_models(train_data, train_split=0.7)
        
        if results:
            print("Training completed:")
            for name, result in results.items():
                status = result.get('status', 'unknown')
                print(f"  {name}: {status}")
            
            # Save to database
            saved = db.save_trained_models(models.models)
            print(f"Models saved: {saved}")
            
            return True
        else:
            print("Training failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    quick_train()