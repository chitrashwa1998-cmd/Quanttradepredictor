#!/usr/bin/env python3
"""
Quick model training script using the loaded sample data
"""

import sys
sys.path.append('.')

from utils.database_adapter import get_trading_database
from models.xgboost_models import QuantTradingModels

def train_models():
    """Train all 7 AI models quickly"""
    try:
        print("Loading data from database...")
        db = get_trading_database()
        
        # Load the main dataset
        data = db.load_ohlc_data("main_dataset")
        
        if data is None or data.empty:
            print("❌ No data found in database")
            return False
        
        print(f"✅ Loaded {len(data)} data points")
        print(f"Data range: {data.index.min()} to {data.index.max()}")
        
        # Initialize model trainer
        print("Initializing model trainer...")
        models = QuantTradingModels()
        
        # Train all models
        print("Training all models...")
        results = models.train_all_models(data, train_split=0.8)
        
        if results:
            print("\n✅ Model Training Results:")
            for model_name, result in results.items():
                if result['status'] == 'success':
                    accuracy = result.get('accuracy', 0)
                    print(f"  {model_name}: {accuracy:.4f} accuracy")
                else:
                    print(f"  {model_name}: Failed - {result.get('error', 'Unknown error')}")
            
            # Save models to database
            print("\nSaving models to database...")
            saved = db.save_trained_models(models.models)
            
            if saved:
                print("✅ Models saved to database successfully")
                return True
            else:
                print("⚠️ Models trained but failed to save to database")
                return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_models()
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n💥 Training failed!")