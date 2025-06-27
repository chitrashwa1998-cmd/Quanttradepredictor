#!/usr/bin/env python3
"""
Quick direction model training script to get predictions working
"""

import streamlit as st
import pandas as pd
import numpy as np
from models.direction_model import DirectionModel
from features.direction_technical_indicators import DirectionTechnicalIndicators
from features.direction_custom_engineered import add_custom_direction_features
from features.direction_lagged_features import add_lagged_direction_features
from features.direction_time_context import add_time_context_features
from utils.database_adapter import get_trading_database

def quick_train_direction_model():
    """Quickly train direction model and save to session state"""
    print("🚀 Starting quick direction model training...")
    
    # Get database connection
    db = get_trading_database()
    
    # Load data
    print("📊 Loading data from database...")
    data = db.recover_data()
    if data is None or data.empty:
        print("❌ No data available. Please upload data first.")
        return False
    
    print(f"✅ Loaded {len(data)} rows of data")
    
    # Calculate direction features
    print("🔧 Calculating direction features...")
    
    # Technical indicators
    tech_indicators = DirectionTechnicalIndicators()
    features_df = tech_indicators.calculate_all_direction_indicators(data.copy())
    
    # Custom engineered features
    features_df = add_custom_direction_features(features_df)
    
    # Lagged features
    features_df = add_lagged_direction_features(features_df)
    
    # Time context features
    features_df = add_time_context_features(features_df)
    
    print(f"✅ Generated {features_df.shape[1]} direction features")
    print(f"Features: {list(features_df.columns[:10])}...")
    
    # Initialize and train direction model
    print("🤖 Training direction model...")
    direction_model = DirectionModel()
    
    # Create target
    target = direction_model.create_target(data)
    print(f"✅ Created target with {len(target)} samples")
    
    # Prepare features for training
    prepared_features = direction_model.prepare_features(features_df)
    print(f"✅ Prepared {prepared_features.shape[1]} features for training")
    
    # Train the model
    results = direction_model.train(prepared_features, target, train_split=0.8)
    print(f"✅ Model trained with accuracy: {results.get('test_accuracy', 'N/A'):.3f}")
    
    # Save to database and session state if we're in Streamlit context
    try:
        import streamlit as st
        
        # Store in session state
        if 'direction_trained_models' not in st.session_state:
            st.session_state.direction_trained_models = {}
        
        st.session_state.direction_trained_models['direction'] = direction_model
        st.session_state.direction_features = features_df
        
        # Also save to database for persistence
        db.save_trained_models({'direction_model': direction_model})
        
        print("✅ Direction model saved to session state and database")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not save to session state (not in Streamlit context): {e}")
        return True

if __name__ == "__main__":
    success = quick_train_direction_model()
    if success:
        print("🎉 Direction model training completed successfully!")
        print("Now you can go to Predictions page and generate direction predictions.")
    else:
        print("❌ Direction model training failed.")