
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any

class TrendSidewaysModel:
    """Trend vs sideways classification model."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'
        self.model_name = 'trend_sideways'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create trend vs sideways target using improved algorithm."""
        # Calculate moving averages for trend detection
        sma_5 = df['Close'].rolling(5).mean()
        sma_10 = df['Close'].rolling(10).mean()
        sma_20 = df['Close'].rolling(20).mean()
        ema_8 = df['Close'].ewm(span=8).mean()
        ema_21 = df['Close'].ewm(span=21).mean()

        # Calculate price momentum (historical only)
        price_change_5 = df['Close'] / df['Close'].shift(5) - 1
        price_change_10 = df['Close'] / df['Close'].shift(10) - 1
        price_change_20 = df['Close'] / df['Close'].shift(20) - 1

        # Calculate volatility for adaptive thresholds
        returns = df['Close'].pct_change()
        volatility_10 = returns.rolling(10).std()
        volatility_20 = returns.rolling(20).std()

        # Data-adaptive trend threshold
        vol_50th = volatility_20.quantile(0.50)
        base_threshold = np.maximum(0.001, vol_50th * 0.8)
        trend_threshold = np.maximum(base_threshold, volatility_20 * 1.0)

        # 1. MOVING AVERAGE TREND STRENGTH
        ema_bullish_trend = (ema_8 > ema_21) & (df['Close'] > ema_8)
        ema_bearish_trend = (ema_8 < ema_21) & (df['Close'] < ema_8)
        ema_trend_strength = ema_bullish_trend | ema_bearish_trend

        # SMA slope analysis
        sma_20_slope = (sma_20 - sma_20.shift(5)) / sma_20.shift(5)
        strong_sma_trend = np.abs(sma_20_slope) > (trend_threshold * 0.5)

        # 2. MOMENTUM ANALYSIS
        momentum_5_strong = np.abs(price_change_5) > trend_threshold
        momentum_10_strong = np.abs(price_change_10) > (trend_threshold * 1.2)
        momentum_20_strong = np.abs(price_change_20) > (trend_threshold * 1.5)

        # Momentum consistency
        momentum_consistent = (
            (price_change_5 > 0) & (price_change_10 > 0) & (price_change_20 > 0)
        ) | (
            (price_change_5 < 0) & (price_change_10 < 0) & (price_change_20 < 0)
        )

        # 3. VOLATILITY REGIME
        volatility_expansion = volatility_10 > (volatility_20 * 1.2)

        # 4. PRICE POSITION
        price_vs_sma20 = df['Close'] / sma_20 - 1
        strong_price_position = np.abs(price_vs_sma20) > (trend_threshold * 0.5)

        # 5. TREND PERSISTENCE
        trend_persistence_3 = (
            ema_trend_strength & 
            ema_trend_strength.shift(1) & 
            ema_trend_strength.shift(2)
        )

        # MULTI-REGIME TREND CLASSIFICATION
        strong_trend_strict = (
            momentum_consistent & 
            (momentum_10_strong | momentum_20_strong) &
            ema_trend_strength &
            strong_price_position
        )

        moderate_trend = (
            ((momentum_10_strong | momentum_20_strong) & ema_trend_strength) |
            (momentum_consistent & strong_sma_trend) |
            (volatility_expansion & strong_price_position & ema_trend_strength)
        ) & ~strong_trend_strict

        weak_trend = (
            (momentum_5_strong & ema_trend_strength) |
            (trend_persistence_3 & (strong_price_position | strong_sma_trend)) |
            (volatility_expansion & (momentum_5_strong | strong_sma_trend))
        ) & ~strong_trend_strict & ~moderate_trend

        # FINAL BINARY CLASSIFICATION
        all_trend_strength = strong_trend_strict.astype(int) * 3 + moderate_trend.astype(int) * 2 + weak_trend.astype(int) * 1

        # Adaptive threshold: aim for 10-20% trending periods
        trend_threshold_percentile = 85  # Top 15% as trending
        trend_cutoff = np.percentile(all_trend_strength, trend_threshold_percentile)

        strong_trend = all_trend_strength >= trend_cutoff

        # Convert to binary: 1 = trending, 0 = sideways
        target = strong_trend.astype(int)

        # Debug information
        trend_counts = target.value_counts()
        print(f"Trend/Sideways Distribution: Trending={trend_counts.get(1, 0)}, Sideways={trend_counts.get(0, 0)}")
        if len(trend_counts) > 0:
            print(f"Trending percentage: {trend_counts.get(1, 0) / len(target) * 100:.1f}%")

        return target

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for trend/sideways model."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        from features.technical_indicators import TechnicalIndicators
        
        # Calculate trend-specific indicators
        result_df = TechnicalIndicators.calculate_trend_indicators(df)
        
        # Define trend-specific features
        trend_features = ['adx', 'rsi', 'bb_width', 'dc_upper', 'dc_lower', 'dc_width', 'ema_fast', 'ema_slow', 'macd_histogram', 'obv']
        
        # Check which features are available
        available_features = [col for col in trend_features if col in result_df.columns]
        
        if len(available_features) == 0:
            raise ValueError(f"No trend features found. Available columns: {list(result_df.columns)}")
        
        # Select only trend features and remove NaN
        result_df = result_df[available_features].dropna()
        
        if result_df.empty:
            raise ValueError("DataFrame is empty after removing NaN values")
        
        print(f"Trend model using {len(available_features)} features: {available_features}")
        
        self.feature_names = available_features
        return result_df

    def train(self, X: pd.DataFrame, y: pd.Series, train_split: float = 0.8) -> Dict[str, Any]:
        """Train trend/sideways classification model."""
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
        """Make predictions using trained trend/sideways model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
