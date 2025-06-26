
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
        
        # Specific features for volatility prediction
        self.volatility_features = [
            'volatility_10', 'atr', 'volatility_regime', 'ema_5', 
            'bb_upper', 'bb_lower', 'bb_width', 'high_low_ratio', 
            'price_vs_vwap', 'momentum_acceleration', 'rsi', 'bb_position'
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
            vol_stats = future_vol.describe()
            print(f"Volatility Target Statistics:")
            print(f"  Count: {vol_stats['count']}")
            print(f"  Mean: {vol_stats['mean']:.6f}")
            print(f"  Min: {vol_stats['min']:.6f}")
            print(f"  Max: {vol_stats['max']:.6f}")

        return future_vol

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for volatility model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("DataFrame is empty after removing NaN values")

        # Filter to only include volatility-specific features
        available_volatility_features = [col for col in self.volatility_features if col in df_clean.columns]
        
        if len(available_volatility_features) == 0:
            raise ValueError(f"None of the specified volatility features found. Available columns: {list(df_clean.columns)}")
        
        result_df = df_clean[available_volatility_features]
        
        print(f"Using {len(available_volatility_features)} specified features for volatility model: {available_volatility_features}")
        
        if len(available_volatility_features) < len(self.volatility_features):
            missing_features = [f for f in self.volatility_features if f not in available_volatility_features]
            print(f"Warning: Missing volatility features: {missing_features}")

        self.feature_names = available_volatility_features
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
        X_scaled = self.scaler.transform(X_features)
        predictions = self.model.predict(X_scaled)

        return predictions, None
