
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from typing import Dict, Tuple, Any

class MagnitudeModel:
    """Magnitude prediction model for predicting the size of price moves."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'regression'
        self.model_name = 'magnitude'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create magnitude target (absolute percentage change)."""
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        magnitude = np.abs(future_return) * 100
        return magnitude

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for magnitude model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_clean = df.dropna()
        if df_clean.empty:
            raise ValueError("DataFrame is empty after removing NaN values")

        # Select feature columns
        feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_cols = [col for col in feature_cols if not col.startswith(('target_', 'future_'))]

        # Remove data leakage features
        leakage_features = [
            'Prediction', 'predicted_direction', 'predictions',
            'Signal', 'Signal_Name', 'Confidence',
            'accuracy', 'precision', 'recall'
        ]
        feature_cols = [col for col in feature_cols if col not in leakage_features]

        if not feature_cols:
            raise ValueError("No feature columns found")

        result_df = df_clean[feature_cols]
        self.feature_names = list(result_df.columns)
        
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train magnitude prediction model."""
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
        """Make predictions using trained magnitude model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions, None
