#!/usr/bin/env python3
"""
Force complete training for all 7 AI models including magnitude and volatility regression
"""

import sys
import os
sys.path.append('.')

from models.xgboost_models import QuantTradingModels
from utils.database_adapter import get_trading_database
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def force_train_regression_models():
    """Force train magnitude and volatility regression models"""
    print("Starting forced training for regression models...")
    
    # Initialize components
    db = get_trading_database()
    model_trainer = QuantTradingModels()
    
    # Load data
    data = db.load_ohlc_data('main_dataset')
    if data is None or data.empty:
        print("No data available")
        return False
    
    print(f"Loaded {len(data)} rows of data")
    
    # Prepare features and targets
    X = model_trainer.prepare_features(data)
    targets = model_trainer.create_targets(data)
    
    # Train magnitude regression model
    if 'magnitude' in targets:
        print("Training magnitude regression model...")
        target = targets['magnitude'].fillna(0.001)
        common_idx = X.index.intersection(target.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.loc[common_idx], target.loc[common_idx], 
            test_size=0.2, random_state=42
        )
        
        # Create and train ensemble
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        
        # Train models
        xgb_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions and calculate metrics
        xgb_pred = xgb_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        ensemble_pred = (xgb_pred + rf_pred) / 2
        
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        # Store model
        model_trainer.models['magnitude'] = {
            'ensemble': xgb_model,  # Use XGBoost as primary
            'metrics': {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)},
            'feature_importance': dict(zip(model_trainer.feature_names, xgb_model.feature_importances_)),
            'task_type': 'regression',
            'predictions': ensemble_pred
        }
        model_trainer.scalers['magnitude'] = scaler
        print(f"Magnitude model trained - MSE: {mse:.6f}")
    
    # Train volatility regression model
    if 'volatility' in targets:
        print("Training volatility regression model...")
        target = targets['volatility'].fillna(0.01)
        common_idx = X.index.intersection(target.index)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.loc[common_idx], target.loc[common_idx], 
            test_size=0.2, random_state=42
        )
        
        # Create and train ensemble
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
        
        # Train models
        xgb_model.fit(X_train_scaled, y_train)
        rf_model.fit(X_train_scaled, y_train)
        
        # Make predictions and calculate metrics
        xgb_pred = xgb_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(X_test_scaled)
        ensemble_pred = (xgb_pred + rf_pred) / 2
        
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        # Store model
        model_trainer.models['volatility'] = {
            'ensemble': xgb_model,  # Use XGBoost as primary
            'metrics': {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)},
            'feature_importance': dict(zip(model_trainer.feature_names, xgb_model.feature_importances_)),
            'task_type': 'regression',
            'predictions': ensemble_pred
        }
        model_trainer.scalers['volatility'] = scaler
        print(f"Volatility model trained - MSE: {mse:.6f}")
    
    # Save all models to database
    print("Saving models to database...")
    model_trainer._save_models_to_database()
    
    print(f"Completed training. Total models: {len(model_trainer.models)}")
    return True

if __name__ == "__main__":
    success = force_train_regression_models()
    if success:
        print("All regression models completed successfully!")
    else:
        print("Training failed")
        sys.exit(1)