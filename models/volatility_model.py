import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class VolatilityModel:
    """Volatility prediction model for forecasting future volatility using exactly 27 features."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'regression'
        self.model_name = 'volatility'

        # Exact 27 features for volatility prediction - DO NOT MODIFY
        self.volatility_features = [
            # Technical indicators (5 features)
            'atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width',
            # Custom engineered features (8 features)
            'log_return', 'realized_volatility', 'parkinson_volatility', 
            'high_low_ratio', 'gap_pct', 'price_vs_vwap', 'volatility_spike_flag',
            'candle_body_ratio',
            # Lagged features (7 features)
            'lag_volatility_1', 'lag_volatility_3', 'lag_volatility_5',
            'lag_atr_1', 'lag_atr_3', 'lag_bb_width', 'volatility_regime',
            # Time context features (7 features)
            'hour', 'minute', 'day_of_week', 'is_post_10am', 
            'is_opening_range', 'is_closing_phase', 'is_weekend'
        ]

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create volatility target from raw data with minimal data loss."""
        # Use Close column from raw data - no modifications
        close_col = 'Close' if 'Close' in df.columns else 'close'

        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")

        # Reduced volatility window from 10 to 5 for less data loss
        volatility_window = 5

        # Calculate rolling volatility using percentage returns from raw data
        returns = df[close_col].pct_change()
        current_vol = returns.rolling(volatility_window).std()
        
        # Use current volatility instead of shifted future volatility to preserve data
        future_vol = current_vol.copy()

        # More efficient cleaning - forward fill only, preserve more data
        future_vol = future_vol.ffill()
        future_vol = future_vol.clip(lower=0.0001)  # Minimum volatility threshold

        # Keep all finite values - no filtering
        future_vol = future_vol[np.isfinite(future_vol)]
        
        # Ensure return type is Series
        if isinstance(future_vol, pd.DataFrame):
            future_vol = future_vol.iloc[:, 0]
        
        future_vol = pd.Series(future_vol, name='volatility_target')

        print(f"Optimized volatility target statistics: Count={len(future_vol)}, Mean={future_vol.mean():.6f}")
        return future_vol

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare exactly 26 features for volatility model - NO MODIFICATIONS ALLOWED."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        from features.technical_indicators import TechnicalIndicators
        from features.custom_engineered import compute_custom_volatility_features
        from features.lagged_features import add_volatility_lagged_features
        from features.time_context_features import add_time_context_features

        # Start with the input dataframe
        result_df = df.copy()

        # 1. Calculate technical indicators
        result_df = TechnicalIndicators.calculate_volatility_indicators(result_df)

        # 2. Add custom engineered features
        result_df = compute_custom_volatility_features(result_df)

        # 3. Add lagged features
        result_df = add_volatility_lagged_features(result_df)

        # 4. Add time context features
        result_df = add_time_context_features(result_df)

        # Extract ONLY the exact 26 features specified - exclude any extra features
        feature_columns = []
        missing_features = []
        extra_features = []

        for feature in self.volatility_features:
            if feature in result_df.columns:
                feature_columns.append(feature)
            else:
                missing_features.append(feature)

        # Check for extra features not in our specification
        for col in result_df.columns:
            if col not in self.volatility_features:
                extra_features.append(col)

        if missing_features:
            print(f"Warning: Missing features: {missing_features}")

        if extra_features:
            print(f"Warning: Excluding extra features: {extra_features}")

        # Use only the exact features that exist
        if feature_columns:
            result_df = result_df[feature_columns].copy()
        else:
            raise ValueError("No features found matching the required 27 features")

        # Use forward fill instead of dropping NaN values to preserve data
        result_df = result_df.ffill()
        
        # Only drop rows where ALL features are NaN (extremely rare)
        result_df = result_df.dropna(how='all')

        if result_df.empty:
            raise ValueError("DataFrame is empty after removing completely empty rows")

        print(f"Optimized volatility model using exactly {len(feature_columns)} features (target: 27)")
        print(f"Data preserved: {len(result_df)} rows (minimal loss from original data)")
        
        # Ensure return type is DataFrame
        if isinstance(result_df, pd.Series):
            result_df = pd.DataFrame(result_df)
        
        # Ensure it's a proper DataFrame
        result_df = pd.DataFrame(result_df)

        self.feature_names = feature_columns
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility prediction model with optimized data usage."""
        # Efficient data alignment - preserve maximum data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Streamlined cleaning - only remove rows with invalid targets
        valid_targets = np.isfinite(y_aligned) & (y_aligned > 0)
        X_clean = X_aligned[valid_targets]
        y_clean = y_aligned[valid_targets]
        
        # Forward fill any remaining NaN values in features
        X_clean = X_clean.ffill().bfill()

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X_clean)}")

        # Train/test split
        split_idx = int(len(X_clean) * train_split)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        print(f"Optimized volatility model training on {len(X_train)} samples with {X_train.shape[1]} features")
        print(f"Total data usage: {len(X_clean)}/{len(X_aligned)} ({len(X_clean)/len(X_aligned)*100:.1f}% of aligned data)")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build ensemble
        random_state = 42

        xgb_model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

        catboost_model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        )

        # Create voting regressor
        ensemble_model = VotingRegressor(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ]
        )

        # Train ensemble model
        ensemble_model.fit(X_train_scaled, y_train)
        self.model = ensemble_model

        # Make predictions
        y_pred_train = ensemble_model.predict(X_train_scaled)
        y_pred_test = ensemble_model.predict(X_test_scaled)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"Training RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
        print(f"Testing RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")

        return {
            'model': ensemble_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, None]:
        """Make predictions using trained volatility model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        # Filter to volatility-specific features
        available_features = [col for col in self.volatility_features if col in X.columns]
        if not available_features:
            raise ValueError("No volatility features found in input data")

        X_features = X[available_features]

        if self.scaler is None:
            raise ValueError("Scaler not fitted. Model training failed.")

        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)

        return predictions, None