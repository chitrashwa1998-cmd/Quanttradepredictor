
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from typing import Dict, Tuple, Any

class TradingSignalModel:
    """Trading signal generation model for buy/sell/hold signals."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.task_type = 'classification'
        self.model_name = 'trading_signal'

    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create trading signal target (buy/sell/hold) using scalping strategy."""
        # Calculate short-term momentum
        price_momentum_1 = df['Close'].shift(-1) / df['Close'] - 1
        price_momentum_2 = df['Close'].shift(-2) / df['Close'] - 1

        # Very short moving averages for scalping
        ema_3 = df['Close'].ewm(span=3).mean()
        ema_9 = df['Close'].ewm(span=9).mean()
        sma_5 = df['Close'].rolling(5).mean()

        # Calculate volatility
        volatility_short = df['Close'].pct_change().rolling(10).std()
        volatility_long = df['Close'].pct_change().rolling(20).std()

        # Scalping thresholds
        base_threshold = 0.0008  # 0.08% base threshold
        volatility_multiplier = 0.2

        dynamic_threshold = np.maximum(base_threshold, volatility_short * volatility_multiplier)
        buy_threshold = dynamic_threshold * 0.7
        sell_threshold = -dynamic_threshold * 0.7

        # Signal criteria
        micro_up = price_momentum_1 > buy_threshold * 0.5
        micro_down = price_momentum_1 < sell_threshold * 0.5

        # EMA crossover signals
        ema_bullish = (df['Close'] > ema_3) & (ema_3 > ema_9)
        ema_bearish = (df['Close'] < ema_3) & (ema_3 < ema_9)

        # Price action signals
        breakout_up = df['Close'] > df['High'].rolling(3).max().shift(1)
        breakout_down = df['Close'] < df['Low'].rolling(3).min().shift(1)

        # Volume confirmation (if available)
        if 'Volume' in df.columns:
            volume_avg = df['Volume'].rolling(10).mean()
            high_volume = df['Volume'] > volume_avg * 1.2
        else:
            high_volume = pd.Series(True, index=df.index)

        # Volatility expansion
        vol_expansion = volatility_short > volatility_long * 1.1

        # Price relative to recent range
        high_5 = df['High'].rolling(5).max()
        low_5 = df['Low'].rolling(5).min()
        range_5 = high_5 - low_5
        price_position = (df['Close'] - low_5) / range_5

        upper_range = price_position > 0.7
        lower_range = price_position < 0.3

        # Convert to boolean Series
        micro_up = micro_up.fillna(False).astype(bool)
        micro_down = micro_down.fillna(False).astype(bool)
        ema_bullish = ema_bullish.fillna(False).astype(bool)
        ema_bearish = ema_bearish.fillna(False).astype(bool)
        high_volume = high_volume.fillna(False).astype(bool)
        breakout_up = breakout_up.fillna(False).astype(bool)
        breakout_down = breakout_down.fillna(False).astype(bool)
        vol_expansion = vol_expansion.fillna(False).astype(bool)
        lower_range = lower_range.fillna(False).astype(bool)
        upper_range = upper_range.fillna(False).astype(bool)

        momentum_2_bullish = (price_momentum_2 > buy_threshold).fillna(False).astype(bool)
        momentum_2_bearish = (price_momentum_2 < sell_threshold).fillna(False).astype(bool)
        price_above_sma = (df['Close'] > sma_5).fillna(False).astype(bool)
        price_below_sma = (df['Close'] < sma_5).fillna(False).astype(bool)

        # SCALPING BUY SIGNALS
        scalp_buy_signals = (
            (micro_up & ema_bullish) |
            (breakout_up) |
            (momentum_2_bullish) |
            (lower_range & micro_up) |
            (ema_bullish & price_above_sma) |
            (micro_up & vol_expansion)
        )

        # SCALPING SELL SIGNALS
        scalp_sell_signals = (
            (micro_down & ema_bearish) |
            (breakout_down) |
            (momentum_2_bearish) |
            (upper_range & micro_down) |
            (ema_bearish & price_below_sma) |
            (micro_down & vol_expansion)
        )

        # Volatility filter
        volatility_quantile = volatility_short.rolling(50).quantile(0.1)
        sufficient_volatility = (volatility_short > volatility_quantile).fillna(True).astype(bool)

        scalp_buy_signals = scalp_buy_signals & sufficient_volatility
        scalp_sell_signals = scalp_sell_signals & sufficient_volatility

        # Ensure no conflicting signals
        scalp_buy_signals = scalp_buy_signals.fillna(False).astype(bool)
        scalp_sell_signals = scalp_sell_signals.fillna(False).astype(bool)

        conflicting = scalp_buy_signals & scalp_sell_signals
        scalp_buy_signals = scalp_buy_signals & ~conflicting
        scalp_sell_signals = scalp_sell_signals & ~conflicting

        # Create final signals
        signals = np.where(scalp_buy_signals, 2, 
                          np.where(scalp_sell_signals, 0, 1))  # 2=Buy, 1=Hold, 0=Sell

        # Force better distribution if too many holds
        signal_counts = pd.Series(signals).value_counts()
        hold_percentage = signal_counts.get(1, 0) / len(signals) * 100

        if hold_percentage > 80:
            price_change_small = df['Close'].pct_change(1).fillna(0)
            additional_buys = (signals == 1) & (price_change_small > 0.0005)
            additional_sells = (signals == 1) & (price_change_small < -0.0005)
            signals = np.where(additional_buys, 2, signals)
            signals = np.where(additional_sells, 0, signals)

        target = pd.Series(signals, index=df.index)

        # Debug information
        signal_counts = pd.Series(signals).value_counts()
        total_signals = len(signals)
        buy_pct = (signal_counts.get(2, 0) / total_signals) * 100
        sell_pct = (signal_counts.get(0, 0) / total_signals) * 100
        hold_pct = (signal_counts.get(1, 0) / total_signals) * 100

        print(f"Trading Signal Distribution:")
        print(f"  Buy: {signal_counts.get(2, 0)} ({buy_pct:.1f}%)")
        print(f"  Hold: {signal_counts.get(1, 0)} ({hold_pct:.1f}%)")
        print(f"  Sell: {signal_counts.get(0, 0)} ({sell_pct:.1f}%)")

        return target

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for trading signal model."""
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
        """Train trading signal generation model."""
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
        """Make predictions using trained trading signal model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities
