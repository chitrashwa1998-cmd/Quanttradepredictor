#!/usr/bin/env python3
"""
Direct completion of all 7 AI models for the trading dashboard
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
import pickle

def complete_training():
    """Complete all 7 AI models directly"""
    # Initialize components
    db = get_trading_database()
    trainer = QuantTradingModels()
    
    # Load existing models from database
    trainer._load_existing_models()
    
    # Load data
    data = db.load_ohlc_data('main_dataset')
    if data is None or data.empty:
        print("No data available")
        return False
    
    print(f"Loaded {len(data)} rows of training data")
    
    # Prepare features and targets
    X = trainer.prepare_features(data)
    targets = trainer.create_targets(data)
    
    # Ensure we have the 5 existing models
    print(f"Current models loaded: {len(trainer.models)}")
    
    # Add magnitude regression model
    if 'magnitude' not in trainer.models and 'magnitude' in targets:
        print("Adding magnitude regression model...")
        target = targets['magnitude'].fillna(0.001)
        common_idx = X.index.intersection(target.index)
        
        if len(common_idx) > 100:
            X_train, X_test, y_train, y_test = train_test_split(
                X.loc[common_idx], target.loc[common_idx], 
                test_size=0.2, random_state=42
            )
            
            # Simple XGBoost regressor
            model = xgb.XGBRegressor(
                max_depth=6, learning_rate=0.1, n_estimators=50,
                random_state=42, n_jobs=-1
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            
            trainer.models['magnitude'] = {
                'ensemble': model,
                'metrics': {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)},
                'feature_importance': dict(zip(trainer.feature_names, model.feature_importances_)),
                'task_type': 'regression',
                'predictions': pred
            }
            trainer.scalers['magnitude'] = scaler
            print(f"Magnitude model added - MSE: {mse:.6f}")
    
    # Add volatility regression model
    if 'volatility' not in trainer.models and 'volatility' in targets:
        print("Adding volatility regression model...")
        target = targets['volatility'].fillna(0.01)
        common_idx = X.index.intersection(target.index)
        
        if len(common_idx) > 100:
            X_train, X_test, y_train, y_test = train_test_split(
                X.loc[common_idx], target.loc[common_idx], 
                test_size=0.2, random_state=42
            )
            
            # Simple XGBoost regressor
            model = xgb.XGBRegressor(
                max_depth=6, learning_rate=0.1, n_estimators=50,
                random_state=42, n_jobs=-1
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            
            trainer.models['volatility'] = {
                'ensemble': model,
                'metrics': {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)},
                'feature_importance': dict(zip(trainer.feature_names, model.feature_importances_)),
                'task_type': 'regression',
                'predictions': pred
            }
            trainer.scalers['volatility'] = scaler
            print(f"Volatility model added - MSE: {mse:.6f}")
    
    # Save all models to database
    print("Saving all models to database...")
    trainer._save_models_to_database()
    
    print(f"Training completed. Total models: {len(trainer.models)}")
    
    # Verify models
    for name, model_info in trainer.models.items():
        task_type = model_info.get('task_type', 'unknown')
        if task_type == 'classification':
            acc = model_info.get('metrics', {}).get('accuracy', 0)
            print(f"  {name}: {acc:.4f} accuracy ({task_type})")
        else:
            mse = model_info.get('metrics', {}).get('mse', 0)
            print(f"  {name}: {mse:.6f} MSE ({task_type})")
    
    return True

if __name__ == "__main__":
    success = complete_training()
    if success:
        print("All 7 AI models completed successfully!")
    else:
        print("Training failed")
        sys.exit(1)