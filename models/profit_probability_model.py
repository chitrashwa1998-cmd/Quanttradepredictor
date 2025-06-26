
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any

class ProfitProbabilityModel:
    """Profit probability prediction model for predicting likelihood of profit."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'
        self.model_name = 'profit_prob'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create profit probability target based on next 5 periods."""
        returns = df['Close'].pct_change().dropna()
        
        # More realistic profit threshold for 5-min scalping
        base_profit_threshold = 0.001  # 0.1% minimum profit target

        # Look ahead only 5 candles (25 minutes for 5-min data)
        future_returns_list = []
        for i in range(5):
            future_return = df['Close'].shift(-i-1) / df['Close'] - 1
            future_returns_list.append(future_return)

        # Get maximum return within 5 periods
        future_returns_df = pd.concat(future_returns_list, axis=1)
        max_future_return = future_returns_df.max(axis=1)

        # Use adaptive threshold based on actual data distribution
        profit_threshold = np.percentile(max_future_return.dropna(), 65)  # Top 35% as profit opportunities
        profit_threshold = max(profit_threshold, base_profit_threshold)

        target = (max_future_return > profit_threshold).astype(int)

        # Debug information
        profit_prob_stats = target.value_counts()
        print(f"Profit Probability Target Distribution: {profit_prob_stats.to_dict()}")
        print(f"Profit threshold used: {profit_threshold:.4f}")

        return target

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for profit probability model."""
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

        # Add candle behavior features
        expected_candle_features = [
            'body_size', 'upper_wick', 'lower_wick', 'total_range', 'body_ratio', 
            'wick_ratio', 'is_bullish', 'candle_strength', 'doji', 'marubozu', 
            'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
            'bull_streak_3', 'bear_streak_2', 'inside_bar', 'outside_bar', 
            'reversal_bar', 'gap_up', 'gap_down', 'direction_change', 
            'momentum_surge', 'minute_of_hour', 'is_opening_range', 'is_closing_phase'
        ]

        for feature in expected_candle_features:
            if feature in df_clean.columns and feature not in feature_cols and feature not in leakage_features:
                feature_cols.append(feature)

        if not feature_cols:
            raise ValueError("No feature columns found")

        result_df = df_clean[feature_cols]
        self.feature_names = list(result_df.columns)
        
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train profit probability prediction model."""
        # Align data
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Clean data
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

        # Remove invalid targets
        valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
        X_clean = X_clean[valid_targets]
        y_clean = y_clean[valid_targets]

        # Ensure we have at least 2 classes
        unique_targets = y_clean.unique()
        if len(unique_targets) < 2:
            raise ValueError(f"Insufficient target classes. Found classes: {unique_targets}")

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

        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1
        )

        self.model = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('catboost', catboost_model),
                ('random_forest', rf_model)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]
        )

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
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
            'probabilities': y_pred_proba,
            'test_indices': X_test.index
        }

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained profit probability model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
