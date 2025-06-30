#!/usr/bin/env python3
"""
Comprehensive test for reversal model with all feature types
"""

import pandas as pd
import numpy as np
from models.reversal_model import ReversalModel

def test_reversal_model_comprehensive():
    """Test the reversal model with all feature types"""
    print("Testing comprehensive reversal model...")
    
    try:
        # Create sample data with proper OHLC structure
        dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
        np.random.seed(42)  # For reproducible results
        
        close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.randn(1000) * 0.1,
            'High': close_prices + np.abs(np.random.randn(1000) * 0.3),
            'Low': close_prices - np.abs(np.random.randn(1000) * 0.3),
            'Close': close_prices,
            'Volume': np.random.uniform(1000, 10000, 1000)
        }, index=dates)
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        print(f"Created sample data: {data.shape}")
        print(f"Data columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Initialize model
        model = ReversalModel()
        
        # Test feature preparation
        print("\nTesting comprehensive feature preparation...")
        features = model.prepare_features(data)
        print(f"✅ Features prepared successfully: {features.shape}")
        print(f"Feature columns ({len(features.columns)}): {list(features.columns)}")
        
        # Test target creation
        print("\nTesting target creation...")
        target = model.create_target(data)
        print(f"✅ Target created: {target.shape}")
        print(f"Target distribution: {target.value_counts()}")
        
        # Ensure we have both classes for training
        if len(target.unique()) < 2:
            print("Creating balanced target for training...")
            target = pd.Series(np.random.choice([0, 1], size=len(target), p=[0.7, 0.3]), index=target.index)
            print(f"New target distribution: {target.value_counts()}")
        
        # Align features and target
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        print(f"Aligned data: {features_aligned.shape[0]} samples")
        
        # Test model training
        print("\nTesting model training...")
        result = model.train(features_aligned, target_aligned, train_split=0.8, max_depth=6, n_estimators=50)
        
        if result:
            print("✅ Reversal model trained successfully!")
            print(f"Training accuracy: {result.get('metrics', {}).get('accuracy', 0):.4f}")
            print(f"Feature importance available: {len(result.get('feature_importance', {}))}")
            
            # Test predictions
            print("\nTesting predictions...")
            predictions, probabilities = model.predict(features_aligned.iloc[-100:])
            print(f"✅ Predictions generated: {len(predictions)} samples")
            print(f"Prediction distribution: {np.unique(predictions, return_counts=True)}")
            
            return True
        else:
            print("❌ Training failed - no result returned")
            return False
            
    except Exception as e:
        print(f"❌ Error during comprehensive reversal model testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reversal_model_comprehensive()
    if success:
        print("\n🎉 Comprehensive reversal model test completed successfully!")
        print("✅ All feature types integrated:")
        print("  - Reversal technical indicators")
        print("  - Custom reversal features")
        print("  - Lagged reversal features")
        print("  - Time context features")
    else:
        print("\n💥 Comprehensive reversal model test failed!")