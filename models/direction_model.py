
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any
import streamlit as st

class DirectionModel:
    """Direction prediction model for predicting price direction (up/down)."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selector = None
        self.selected_features = []
        self.task_type = 'classification'
        self.model_name = 'direction'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create direction target (up/down) - simple version without noise filtering."""
        # Ensure we're working with the Close column only (numeric data)
        close_prices = pd.to_numeric(df['Close'], errors='coerce')
        
        # Simple direction prediction (up/down)
        future_return = close_prices.shift(-1) / close_prices - 1
        
        # Create direction target: 1 for up, 0 for down
        direction_raw = np.where(future_return > 0, 1, 0)
        direction_series = pd.Series(direction_raw, index=df.index)
        
        # Remove NaN values and ensure all values are numeric
        target = direction_series.dropna()
        target = pd.to_numeric(target, errors='coerce').dropna().astype(int)
        
        print(f"Direction target created: {len(target)} samples")
        return target

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train direction model from raw OHLC data"""
        try:
            print(f"🚀 Training direction model with {len(data)} data points")
            
            # Prepare features and target
            features_df = self.prepare_features(data)
            target = self.create_target(data)
            
            # Train the model
            result = self.train(features_df, target)
            
            # Save model to database
            from utils.database_adapter import DatabaseAdapter
            db = DatabaseAdapter()
            db.save_trained_model('direction', self.model, self.scaler, self.feature_names)
            
            return {
                'success': True,
                'model_type': 'direction',
                'accuracy': result.get('test_accuracy', 0.0),
                'training_samples': len(features_df),
                'message': 'Direction model trained successfully'
            }
            
        except Exception as e:
            print(f"Error training direction model: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_type': 'direction'
            }

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for direction model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        from features.direction_technical_indicators import DirectionTechnicalIndicators
        
        # Calculate all direction-specific indicators
        calc = DirectionTechnicalIndicators()
        result_df = calc.calculate_direction_features(df)
        
        # Define core direction-specific features to ensure we have them
        core_direction_features = ['ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd_histogram', 
                                 'bollinger_band_position', 'bb_width', 'stochastic_k', 'stochastic_d', 
                                 'adx', 'obv', 'donchian_high_20', 'donchian_low_20']
        
        # Get all available features excluding OHLC and timestamp columns
        ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Timestamp', 'Date', 'DateTime', 'Time']
        exclude_cols = ohlc_cols + timestamp_cols
        
        # Filter out non-numeric columns by checking data types
        available_features = []
        for col in result_df.columns:
            if col in exclude_cols:
                continue
            # Check if column has datetime or object dtype
            if pd.api.types.is_datetime64_any_dtype(result_df[col]):
                print(f"Excluding datetime column: {col}")
                continue
            if pd.api.types.is_object_dtype(result_df[col]):
                # Try to convert to numeric - if it fails, exclude it
                try:
                    pd.to_numeric(result_df[col], errors='raise')
                    available_features.append(col)
                except (ValueError, TypeError):
                    print(f"Excluding non-numeric object column: {col}")
                    continue
            else:
                available_features.append(col)
        
        if len(available_features) == 0:
            raise ValueError(f"No direction features found. Available columns: {list(result_df.columns)}")
        
        # Select only numeric direction features
        feature_df = result_df[available_features].copy()
        
        # Convert all features to numeric, replacing any non-numeric values with NaN
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        
        # Remove rows with NaN values
        feature_df = feature_df.dropna()
        
        if feature_df.empty:
            raise ValueError("DataFrame is empty after removing NaN values")
        
        print(f"Direction model using {len(available_features)} features: {available_features}")
        
        return feature_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8, max_depth: int = 6, n_estimators: int = 100) -> Dict[str, Any]:
        """Train direction prediction model."""
        # Ensure X and y have the same index
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Remove NaN values
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

        # Remove invalid target values
        valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]

        # Ensure we have at least 2 classes
        unique_targets = y_clean.unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Insufficient target classes. Found classes: {unique_targets}")

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(X_clean)}")

        # Stratified split for balanced classes
        try:
            split_idx = int(len(X_clean) * train_split)
            X_temp_train = X_clean.iloc[:split_idx]
            y_temp_train = y_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_test = y_clean.iloc[split_idx:]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp_train, y_temp_train, 
                test_size=0.1, 
                stratify=y_temp_train, 
                random_state=42
            )
            # Combine validation back to training
            X_train = pd.concat([X_train, X_val])
            y_train = pd.concat([y_train, y_val])
        except ValueError:
            # Fallback to original split if stratification fails
            split_idx = int(len(X_clean) * train_split)
            X_train = X_clean.iloc[:split_idx]
            X_test = X_clean.iloc[split_idx:]
            y_train = y_clean.iloc[:split_idx]
            y_test = y_clean.iloc[split_idx:]

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Exclude OHLC columns and timestamp columns from training features
        ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Timestamp', 'Date', 'Datetime', 'Time']
        exclude_cols = ohlc_cols + timestamp_cols
        
        # Check for and remove any non-numeric columns before training
        numeric_columns = []
        for col in X_train.columns:
            if col in exclude_cols:
                print(f"Excluding OHLC/timestamp column from training: {col}")
                continue
            try:
                # Try to convert to numeric - if successful, it's a valid feature
                pd.to_numeric(X_train[col].iloc[:10], errors='raise')
                numeric_columns.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column from training: {col}")
        
        # Use only numeric direction features (excluding OHLC and timestamp)
        X_train_selected = X_train[numeric_columns].values
        X_test_selected = X_test[numeric_columns].values
        
        # Store selected feature names
        self.selected_features = numeric_columns
        self.feature_names = numeric_columns  # Also store as feature_names for compatibility
        print(f"Using {len(self.selected_features)} numeric direction features")
        print(f"Features: {self.selected_features}")
        
        # Standard scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)

        # Build ensemble model
        random_state = 42

        # XGBoost Classifier with configurable parameters
        xgb_model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=0.05,
            n_estimators=n_estimators,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )

        # CatBoost Classifier
        catboost_model = CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=128,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False
        )

        # Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )

        # Create voting classifier (ensemble without calibration)
        self.model = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        

        # Feature importance from XGBoost in ensemble
        feature_importance = {}
        try:
            if hasattr(self.model, 'named_estimators_'):
                xgb_estimator = self.model.named_estimators_['xgboost']
                feature_importance = dict(zip(self.selected_features, xgb_estimator.feature_importances_))
        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return {
            'model': self.model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'feature_names': self.selected_features,
            'task_type': self.task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained direction model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.selected_features is None:
            raise ValueError("No selected features available. Model not properly trained.")

        # Validate features
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        # Check if all required features are present
        missing_features = [f for f in self.selected_features if f not in X.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Use only the selected features that were used during training
        X_selected = X[self.selected_features].values
        X_scaled = self.scaler.transform(X_selected)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
