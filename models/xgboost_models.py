import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor

from .model_manager import ModelManager

# Backward compatibility - use ModelManager as QuantTradingModels
class QuantTradingModels(ModelManager):
    """Volatility-only trading model using XGBoost, CatBoost, and Random Forest."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained volatility model from database if available."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            loaded_models = db.load_trained_models()

            if loaded_models and 'volatility' in loaded_models:
                model_data = loaded_models['volatility']
                # Ensure task_type is present
                if 'task_type' not in model_data:
                    model_data['task_type'] = 'regression'

                self.models = {'volatility': model_data}
                print(f"Loaded volatility model from database")

                # Extract feature names
                if 'feature_names' in model_data and model_data['feature_names']:
                    self.feature_names = model_data['feature_names']
                    print(f"Feature names loaded from volatility model: {len(self.feature_names)} features")
            else:
                print("No volatility model found in database")

        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def _save_models_to_database(self):
        """Save trained volatility model to database for persistence."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()

            # Prepare volatility model for saving
            models_to_save = {}
            if 'volatility' in self.models and 'ensemble' in self.models['volatility']:
                models_to_save['volatility'] = {
                    'ensemble': self.models['volatility']['ensemble'],
                    'feature_names': self.feature_names,
                    'task_type': 'regression'
                }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved volatility model to database with feature names")
                    print(f"Feature names saved: {len(self.feature_names)} features")
                else:
                    print("Failed to save volatility model to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")

    def prepare_features(self, df: pd.DataFrame, model_name: str = 'volatility') -> pd.DataFrame:
        """Prepare features for volatility model training."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Remove any rows with NaN values
        df_clean = df.dropna()

        if df_clean.empty:
            raise ValueError("DataFrame is empty after removing NaN values")

        # Use volatility-specific features only
        volatility_features = ['atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width']

        # Check which features are available
        available_features = [col for col in volatility_features if col in df_clean.columns]

        if len(available_features) == 0:
            raise ValueError(f"No volatility features found. Available columns: {list(df_clean.columns)}")

        result_df = df_clean[available_features]

        if result_df.empty:
            raise ValueError("Feature DataFrame is empty after column selection")

        # Store feature names
        self.feature_names = list(result_df.columns)
        print(f"Volatility model prepared with {len(self.feature_names)} features: {self.feature_names}")

        return result_df

    def create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create target variable for volatility prediction."""
        targets = {}

        # Volatility forecasting (next period volatility)
        volatility_window = 10

        # Calculate rolling volatility using percentage returns for better scaling
        returns = df['Close'].pct_change()
        current_vol = returns.rolling(volatility_window).std()
        future_vol = current_vol.shift(-1)

        # Remove NaN values and ensure we have valid volatility data
        future_vol = future_vol.fillna(method='ffill').fillna(method='bfill')

        # Ensure volatility is positive and finite
        future_vol = future_vol.clip(lower=0.0001)  # Minimum volatility threshold
        future_vol = future_vol[np.isfinite(future_vol)]

        # Debug volatility distribution
        if len(future_vol) > 0:
            vol_stats = future_vol.describe()
            print(f"Volatility Target Statistics:")
            print(f"  Count: {vol_stats['count']}")
            print(f"  Mean: {vol_stats['mean']:.6f}")
            print(f"  Std: {vol_stats['std']:.6f}")
            print(f"  Min: {vol_stats['min']:.6f}")
            print(f"  Max: {vol_stats['max']:.6f}")

        targets['volatility'] = future_vol

        return targets

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, task_type: str = 'regression', train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility model using ensemble of algorithms."""

        if model_name != 'volatility':
            raise ValueError("Only volatility model is supported")

        # Define random state
        random_state = 42

        # Ensure X and y have the same index for proper alignment
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Remove NaN values and ensure we have valid targets
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

        # For regression tasks, remove NaN and infinite values
        valid_targets = np.isfinite(y_clean) & (y_clean > 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]

        if len(y_clean) == 0:
            raise ValueError(f"No valid target values for volatility after cleaning")

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X_clean)}")

        # Train/test split
        split_idx = int(len(X_clean) * train_split)
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        print(f"Training volatility model on {len(X_train)} samples ({len(X_train)/len(X_clean)*100:.1f}%), testing on {len(X_test)} samples ({len(X_test)/len(X_clean)*100:.1f}%)")

        # Standard scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store scaler
        self.scalers['volatility'] = scaler

        # Regression ensemble: XGBoost + CatBoost + Random Forest
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

        # Make predictions
        y_pred = ensemble_model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }

        # Calculate individual model scores for comparison
        individual_scores = {}
        for name, model in ensemble_model.named_estimators_.items():
            individual_pred = model.predict(X_test_scaled)
            individual_scores[f'{name}_mse'] = mean_squared_error(y_test, individual_pred)
            individual_scores[f'{name}_mae'] = mean_absolute_error(y_test, individual_pred)

        metrics.update(individual_scores)

        # Get feature importance (use XGBoost as primary)
        try:
            xgb_estimator = ensemble_model.named_estimators_['xgboost']
            feature_importance = dict(zip(self.feature_names, xgb_estimator.feature_importances_))
            print(f"Feature importance extracted for volatility: {len(feature_importance)} features")

            # Debug: Show feature importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print(f"Volatility feature importance: {sorted_importance}")

        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            feature_importance = {}

        # Store model
        self.models['volatility'] = {
            'ensemble': ensemble_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.feature_names,
            'task_type': 'regression',
            'predictions': y_pred,
            'test_indices': X_test.index,
            'ensemble_type': 'voting_regressor',
            'base_models': list(ensemble_model.named_estimators_.keys())
        }

        return self.models['volatility']

    def train_all_models(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility model."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features
        status_text.text("Preparing features...")
        X = self.prepare_features(df)

        # Create targets
        status_text.text("Creating target variables...")
        targets = self.create_targets(df)

        results = {}

        status_text.text("Training volatility model...")

        try:
            if 'volatility' in targets:
                # Ensure X and target are properly aligned by using common index
                target_series = targets['volatility']
                common_index = X.index.intersection(target_series.index)

                if len(common_index) == 0:
                    st.warning(f"⚠️ No common indices between features and volatility target")
                    results['volatility'] = None
                else:
                    X_aligned = X.loc[common_index]
                    y_aligned = target_series.loc[common_index]

                    result = self.train_model('volatility', X_aligned, y_aligned, 'regression', train_split)
                    results['volatility'] = result
                    st.success(f"✅ Volatility model trained successfully")
            else:
                st.warning(f"⚠️ Volatility target not found")
        except Exception as e:
            st.error(f"❌ Error training volatility model: {str(e)}")
            results['volatility'] = None

        progress_bar.progress(1.0)

        status_text.text("Saving trained model to database...")
        # Automatically save trained model for persistence
        self._save_models_to_database()

        status_text.text("Volatility model trained and saved!")
        return results

    def train_selected_models(self, df: pd.DataFrame, selected_models: list, train_split: float = 0.8) -> Dict[str, Any]:
        """Train volatility model if selected."""
        if 'volatility' not in selected_models:
            st.warning("Only volatility model is available")
            return {}

        return self.train_all_models(df, train_split)

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained volatility model."""
        if model_name != 'volatility':
            raise ValueError(f"Only volatility model is available")

        if 'volatility' not in self.models:
            raise ValueError(f"Volatility model not found. Available models: {list(self.models.keys())}")

        model_info = self.models['volatility']
        model = model_info.get('ensemble') or model_info.get('model')

        # Validate input features
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        # Use volatility-specific features
        volatility_features = ['atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width']
        available_features = [col for col in volatility_features if col in X.columns]

        if len(available_features) == 0:
            raise ValueError(f"No volatility features found in prediction data. Available columns: {list(X.columns)}")

        # Keep only the specified features
        X_features = X[available_features]
        print(f"Using {len(available_features)} volatility features for prediction: {available_features}")

        # Handle scaling
        if 'volatility' in self.scalers:
            try:
                X_scaled = self.scalers['volatility'].transform(X_features)
            except Exception as e:
                expected_features = getattr(self.scalers['volatility'], 'n_features_in_', 'unknown')
                raise ValueError(f"Feature shape mismatch for volatility: expected {expected_features} features, got {X_features.shape[1]}. Please retrain the model with current data.")
        else:
            X_scaled = X_features.values

        # Make predictions using ensemble
        predictions = model.predict(X_scaled)

        # Return predictions (no probabilities for regression)
        return predictions, None

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for volatility model."""
        if model_name != 'volatility':
            print(f"Only volatility model is available")
            return {}

        if 'volatility' not in self.models:
            print(f"Volatility model not found in available models")
            return {}

        model_info = self.models['volatility']
        feature_importance = model_info.get('feature_importance', {})

        print(f"Getting feature importance for volatility: {len(feature_importance)} features")
        return feature_importance