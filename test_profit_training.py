#!/usr/bin/env python3
"""
Test profit probability model training with fixes
"""

import streamlit as st
import pandas as pd
import numpy as np
from features.profit_probability_technical_indicators import ProfitProbabilityTechnicalIndicators
from models.profit_probability_model import ProfitProbabilityModel

def test_profit_probability_training():
    """Test the profit probability model training"""
    
    # Load data from database
    from utils.database_adapter import DatabaseAdapter
    db = DatabaseAdapter()
    
    print("Loading data from database...")
    df = db.load_ohlc_data("main_dataset")
    
    if df is None or len(df) == 0:
        print("❌ No data available in database")
        return False
    
    print(f"✅ Loaded {len(df)} rows from database")
    
    # Calculate profit probability features
    print("🔧 Calculating profit probability features...")
    try:
        profit_prob_features = ProfitProbabilityTechnicalIndicators.calculate_all_profit_probability_indicators(df)
        print(f"✅ Generated {len(profit_prob_features.columns)} profit probability features")
        
        # Initialize and train model
        print("🎯 Training profit probability model...")
        profit_prob_model = ProfitProbabilityModel()
        
        # Create target
        profit_prob_target = profit_prob_model.create_target(df)
        print(f"✅ Created target with {len(profit_prob_target)} samples")
        
        # Train model
        training_result = profit_prob_model.train(profit_prob_features, profit_prob_target)
        print(f"✅ Training completed with accuracy: {training_result.get('accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_profit_probability_training()
    if success:
        print("✅ Profit probability model training test passed!")
    else:
        print("❌ Profit probability model training test failed!")