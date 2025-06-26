
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
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
        """Create direction target (up/down) with noise filtering."""
        # Direction prediction (up/down) - Enhanced with noise filtering
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        
        # Calculate dynamic threshold based on recent volatility
        rolling_vol = df['Close'].pct_change().rolling(20).std()
        noise_threshold = rolling_vol * 0.3  # 30% of recent volatility
        
        # Only predict direction for moves above noise threshold
        significant_up = future_return > noise_threshold
        significant_down = future_return < -noise_threshold
        
        # Create direction target: 1 for up, 0 for down, exclude sideways moves
        direction_raw = np.where(significant_up, 1, np.where(significant_down, 0, np.nan))
        direction_series = pd.Series(direction_raw, index=df.index)
        
        # Remove NaN values (sideways moves)
        target = direction_series.dropna().astype(int)
        
        print(f"Direction target filtering: {len(target)} significant moves out of {len(df)} total ({len(target)/len(df)*100:.1f}%)")
        return target

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for direction model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        from features.technical_indicators import TechnicalIndicators
        
        # Calculate direction-specific indicators
        result_df = TechnicalIndicators.calculate_direction_indicators(df)
        
        # Define direction-specific features
        direction_features = ['rsi', 'macd', 'macd_signal', 'ema_fast', 'ema_slow', 'adx', 'obv', 'stoch_k', 'stoch_d']
        
        # Check which features are available
        available_features = [col for col in direction_features if col in result_df.columns]
        
        if len(available_features) == 0:
            raise ValueError(f"No direction features found. Available columns: {list(result_df.columns)}")
        
        # Select only direction features and remove NaN
        result_df = result_df[available_features].dropna()
        
        if result_df.empty:
            raise ValueError("DataFrame is empty after removing NaN values")
        
        print(f"Direction model using {len(available_features)} features: {available_features}")
        
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
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

        # Feature selection for direction model
        self.selector = SelectKBest(score_func=mutual_info_classif, k=min(35, X_train.shape[1]))
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)
        
        # Store selected feature names
        selected_feature_mask = self.selector.get_support()
        self.selected_features = [X_train.columns[i] for i in range(len(X_train.columns)) if selected_feature_mask[i]]
        print(f"Selected {len(self.selected_features)} most informative features")
        
        # RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)

        # Build ensemble model
        random_state = 42

        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
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

        # Create voting classifier
        base_ensemble = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )
        
        # Apply calibration to reduce overconfidence
        print("Applying calibration to direction model...")
        self.model = CalibratedClassifierCV(
            base_ensemble, 
            method="sigmoid",
            cv=3
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

        # Calculate calibration metrics
        try:
            from sklearn.calibration import calibration_curve
            prob_pos = y_pred_proba[:, 1]
            brier_score = np.mean((prob_pos - y_test) ** 2)
            
            metrics['calibration'] = {
                'brier_score': brier_score,
                'is_calibrated': True,
                'calibration_method': 'Platt Scaling (Sigmoid)',
                'mean_predicted_probability': np.mean(prob_pos),
                'actual_positive_rate': np.mean(y_test)
            }
        except Exception as e:
            print(f"Error calculating calibration metrics: {str(e)}")

        # Feature importance
        feature_importance = {}
        try:
            if hasattr(self.model, 'calibrated_classifiers_'):
                calibrated_classifier = self.model.calibrated_classifiers_[0]
                base_estimator = calibrated_classifier.estimator
                xgb_estimator = base_estimator.named_estimators_['xgboost']
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

        # Validate features
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        # Apply feature selection
        if self.selector is None:
            raise ValueError("Feature selector not initialized")

        X_selected = self.selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
