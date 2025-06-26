
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from typing import Dict, Tuple, Any

class VolatilityModel:
    """Volatility prediction model for forecasting future volatility."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'regression'
        self.model_name = 'volatility'
        
        # Exact 26 features for volatility prediction - DO NOT MODIFY
        self.volatility_features = [
            # Technical indicators (5 features)
            'atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width',
            # Custom engineered features (7 features)
            'log_return', 'realized_volatility', 'parkinson_volatility', 
            'high_low_ratio', 'gap_pct', 'price_vs_vwap', 'volatility_spike_flag',
            # Lagged features (7 features)
            'lag_volatility_1', 'lag_volatility_3', 'lag_volatility_5',
            'lag_atr_1', 'lag_atr_3', 'lag_bb_width', 'volatility_regime',
            # Time context features (7 features)
            'hour', 'minute', 'day_of_week', 'is_post_10am', 
            'is_opening_range', 'is_closing_phase', 'is_weekend'
        ]

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create volatility target (next period volatility)."""
        volatility_window = 10

        # Calculate rolling volatility using percentage returns
        returns = df['Close'].pct_change()
        current_vol = returns.rolling(volatility_window).std()
        future_vol = current_vol.shift(-1)

        # Clean volatility data
        future_vol = future_vol.fillna(method='ffill').fillna(method='bfill')
        future_vol = future_vol.clip(lower=0.0001)  # Minimum volatility threshold
        future_vol = future_vol[np.isfinite(future_vol)]

        # Debug volatility distribution
        if len(future_vol) > 0:
            print(f"Volatility Target Statistics:")
            print(f"  Count: {len(future_vol)}")
            print(f"  Mean: {future_vol.mean():.6f}")
            print(f"  Min: {future_vol.min():.6f}")
            print(f"  Max: {future_vol.max():.6f}")

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
        
        # Extract ONLY the exact 26 features specified
        feature_columns = []
        missing_features = []
        
        for feature in self.volatility_features:
            if feature in result_df.columns:
                feature_columns.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print(f"Available features: {list(result_df.columns)}")
        
        # Use only the exact 26 features that exist
        result_df = result_df[feature_columns].copy()
        
        # Remove rows with any NaN values
        result_df = result_df.dropna()
        
        if result_df.empty:
            raise ValueError("DataFrame is empty after removing NaN values")
        
        print(f"Volatility model using exactly {len(feature_columns)} features (target: 26)")
        print(f"Features: {feature_columns}")
        
        self.feature_names = feature_columns
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility prediction model."""
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Clean data
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

        # Remove invalid targets
        valid_targets = np.isfinite(y_clean) & (y_clean > 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X_clean)}")

        # Train/test split
        split_idx = int(len(X_clean) * train_split)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        print(f"Volatility model training on {len(X_train)} samples with {X_train.shape[1]} features")

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

        self.model = VotingRegressor(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ]
        )

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }

        # Feature importance
        feature_importance = {}
        try:
            xgb_estimator = self.model.named_estimators_['xgboost']
            feature_importance = dict(zip(self.feature_names, xgb_estimator.feature_importances_))
            
            # Debug: Show feature importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 5 volatility features: {sorted_importance[:5]}")
        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return {
            'model': self.model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': self.task_type,
            'predictions': y_pred,
            'test_indices': X_test.index
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
