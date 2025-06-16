import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Any
import streamlit as st
from datetime import datetime

class QuantTradingModels:
    """XGBoost models for quantitative trading predictions."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        # Remove any rows with NaN values
        df_clean = df.dropna()

        # Select feature columns (exclude OHLC and target columns)
        feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_cols = [col for col in feature_cols if not col.startswith(('target_', 'future_'))]

        self.feature_names = feature_cols
        return df_clean[feature_cols]

    def create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create target variables for different prediction tasks."""
        targets = {}

        # 1. Direction prediction (up/down)
        future_return = df['Close'].shift(-1) / df['Close'] - 1
        targets['direction'] = (future_return > 0).astype(int)

        # 2. Magnitude of move (percentage change)
        targets['magnitude'] = np.abs(future_return) * 100

        # 3. Probability of profit (based on next 5 periods only)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std()

        # Profit threshold for 5-candle lookout (25 minutes for 5-min data)
        base_profit_threshold = 0.002  # 0.2% minimum profit target for shorter timeframe
        volatility_adjusted_threshold = max(base_profit_threshold, volatility * 1.0)

        # Look ahead only 5 candles (25 minutes for 5-min data)
        future_returns_list = []
        for i in range(5):
            future_return = df['Close'].shift(-i-1) / df['Close'] - 1
            future_returns_list.append(future_return)

        # Get maximum return within 5 periods
        future_returns_df = pd.concat(future_returns_list, axis=1)
        max_future_return = future_returns_df.max(axis=1)

        # Use a more adaptive threshold based on data volatility
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std()
        profit_threshold = min(0.005, volatility)  # Use 0.5% or data volatility, whichever is smaller

        targets['profit_prob'] = (max_future_return > profit_threshold).astype(int)

        # 4. Volatility forecasting (next period volatility)
        volatility_window = 10
        current_vol = df['Close'].rolling(volatility_window).std()
        future_vol = current_vol.shift(-1)

        # Remove NaN values and ensure we have valid volatility data
        future_vol = future_vol.fillna(method='ffill').fillna(method='bfill')

        # Ensure volatility is positive and finite
        future_vol = future_vol.clip(lower=0.0001)  # Minimum volatility threshold
        future_vol = future_vol[np.isfinite(future_vol)]

        targets['volatility'] = future_vol

        # 5. Trend vs sideways classification
        # Use multiple timeframes to determine trend strength
        price_change_3 = df['Close'].shift(-3) / df['Close'] - 1
        price_change_5 = df['Close'].shift(-5) / df['Close'] - 1
        price_change_10 = df['Close'].shift(-10) / df['Close'] - 1

        # Calculate moving averages for trend detection
        sma_short = df['Close'].rolling(5).mean()
        sma_long = df['Close'].rolling(20).mean()

        # Calculate volatility for adaptive threshold
        volatility = df['Close'].pct_change().rolling(20).std()
        adaptive_threshold = volatility * 2  # Dynamic threshold based on volatility
        base_threshold = 0.015  # 1.5% base threshold

        # Combine thresholds
        trend_threshold = np.maximum(adaptive_threshold, base_threshold)

        # Multiple trend criteria
        # 1. Price momentum over different periods
        strong_trend_3 = np.abs(price_change_3) > trend_threshold
        strong_trend_5 = np.abs(price_change_5) > trend_threshold
        strong_trend_10 = np.abs(price_change_10) > trend_threshold * 1.5

        # 2. Moving average trend
        ma_trend = np.abs(sma_short / sma_long - 1) > 0.01  # 1% difference in MAs

        # 3. Price relative to moving averages
        price_above_ma = df['Close'] > sma_short
        consistent_trend = (price_above_ma == price_above_ma.shift(3)) & (price_above_ma == price_above_ma.shift(5))

        # Combine all criteria - trending if any strong trend indicator is true
        is_trending = (strong_trend_3 | strong_trend_5 | strong_trend_10 | ma_trend | consistent_trend)

        # Convert to binary: 1 = trending, 0 = sideways
        targets['trend_sideways'] = is_trending.astype(int)

        # Debug information for trend_sideways
        trend_counts = targets['trend_sideways'].value_counts()
        print(f"Trend/Sideways Distribution: Trending={trend_counts.get(1, 0)}, Sideways={trend_counts.get(0, 0)}")
        if len(trend_counts) > 0:
            print(f"Trending percentage: {trend_counts.get(1, 0) / len(targets['trend_sideways']) * 100:.1f}%")

        # 6. Reversal points - Enhanced detection
        # Calculate momentum and trend indicators for reversal detection
        price_change_1 = df['Close'].pct_change(1)
        price_change_3 = df['Close'].pct_change(3)
        price_change_5 = df['Close'].pct_change(5)
        
        # Calculate RSI-like momentum indicator
        momentum_window = 14
        gains = price_change_1.where(price_change_1 > 0, 0).rolling(momentum_window).mean()
        losses = (-price_change_1.where(price_change_1 < 0, 0)).rolling(momentum_window).mean()
        momentum_ratio = gains / (losses + 1e-10)  # Avoid division by zero
        momentum_index = 100 - (100 / (1 + momentum_ratio))
        
        # Calculate moving averages for trend context
        sma_short = df['Close'].rolling(5).mean()
        sma_medium = df['Close'].rolling(10).mean()
        sma_long = df['Close'].rolling(20).mean()
        
        # Price position relative to recent highs/lows
        high_10 = df['High'].rolling(10).max()
        low_10 = df['Low'].rolling(10).min()
        price_position = (df['Close'] - low_10) / (high_10 - low_10 + 1e-10)
        
        # Volatility for adaptive thresholds
        volatility = df['Close'].pct_change().rolling(20).std()
        
        # Future price movement for reversal confirmation (look ahead 3-5 periods)
        future_return_3 = df['Close'].shift(-3) / df['Close'] - 1
        future_return_5 = df['Close'].shift(-5) / df['Close'] - 1
        
        # BULLISH REVERSAL CONDITIONS
        # 1. Price at or near recent lows
        near_lows = price_position <= 0.3
        
        # 2. Oversold momentum
        oversold = momentum_index <= 30
        
        # 3. Recent downward momentum
        recent_decline = (price_change_1 < 0) & (price_change_3 < -0.005)
        
        # 4. Moving average support
        ma_support = (df['Close'] <= sma_short) & (sma_short <= sma_medium)
        
        # 5. Future upward movement (confirmation)
        future_bounce = future_return_3 > volatility * 1.5
        
        # BEARISH REVERSAL CONDITIONS
        # 1. Price at or near recent highs
        near_highs = price_position >= 0.7
        
        # 2. Overbought momentum
        overbought = momentum_index >= 70
        
        # 3. Recent upward momentum
        recent_rally = (price_change_1 > 0) & (price_change_3 > 0.005)
        
        # 4. Moving average resistance
        ma_resistance = (df['Close'] >= sma_short) & (sma_short >= sma_medium)
        
        # 5. Future downward movement (confirmation)
        future_decline = future_return_3 < -volatility * 1.5
        
        # Combine conditions for reversal detection
        bullish_reversal = (
            (near_lows & oversold & future_bounce) |
            (recent_decline & ma_support & future_bounce) |
            (oversold & recent_decline & future_bounce)
        )
        
        bearish_reversal = (
            (near_highs & overbought & future_decline) |
            (recent_rally & ma_resistance & future_decline) |
            (overbought & recent_rally & future_decline)
        )
        
        # Final reversal signal: 1 = reversal expected, 0 = no reversal
        reversal_signal = (bullish_reversal | bearish_reversal).astype(int)
        targets['reversal'] = reversal_signal
        
        # Debug information for reversal detection
        reversal_counts = reversal_signal.value_counts()
        total_points = len(reversal_signal)
        reversal_pct = (reversal_counts.get(1, 0) / total_points) * 100 if total_points > 0 else 0
        
        print(f"Reversal Detection Results:")
        print(f"  Total data points: {total_points}")
        print(f"  Reversal signals: {reversal_counts.get(1, 0)} ({reversal_pct:.1f}%)")
        print(f"  No reversal: {reversal_counts.get(0, 0)} ({100-reversal_pct:.1f}%)")
        print(f"  Bullish reversals detected: {bullish_reversal.sum()}")
        print(f"  Bearish reversals detected: {bearish_reversal.sum()}")
        
        # Store additional reversal details for analysis
        if hasattr(self, 'reversal_details'):
            self.reversal_details = {
                'bullish_reversals': bullish_reversal,
                'bearish_reversals': bearish_reversal,
                'momentum_index': momentum_index,
                'price_position': price_position
            }

        # 7. Buy/Sell/Hold signals - SCALPING STRATEGY FOR 5-MIN CANDLES
        # More aggressive signal generation with tighter thresholds

        # Calculate short-term momentum for scalping
        price_momentum_1 = df['Close'].shift(-1) / df['Close'] - 1  # Next candle
        price_momentum_2 = df['Close'].shift(-2) / df['Close'] - 1  # 2 candles ahead
        price_momentum_3 = df['Close'].shift(-3) / df['Close'] - 1  # 3 candles ahead

        # Very short moving averages for scalping
        ema_3 = df['Close'].ewm(span=3).mean()  # 15 min
        ema_9 = df['Close'].ewm(span=9).mean()  # 45 min
        sma_5 = df['Close'].rolling(5).mean()   # 25 min

        # Calculate intraday volatility (more responsive)
        volatility_short = df['Close'].pct_change().rolling(10).std()
        volatility_long = df['Close'].pct_change().rolling(20).std()

        # SCALPING THRESHOLDS - Much tighter for 5-min candles
        base_threshold = 0.0015  # 0.15% base threshold for scalping
        volatility_multiplier = 0.3  # Lower multiplier for tighter signals

        # Dynamic thresholds based on recent volatility
        dynamic_threshold = np.maximum(base_threshold, volatility_short * volatility_multiplier)
        buy_threshold = dynamic_threshold
        sell_threshold = -dynamic_threshold

        # SCALPING SIGNAL CRITERIA

        # 1. Micro momentum signals (very short-term)
        micro_up = price_momentum_1 > buy_threshold * 0.5  # Even smaller moves
        micro_down = price_momentum_1 < sell_threshold * 0.5

        # 2. EMA crossover signals (fast scalping indicator)
        ema_bullish = (df['Close'] > ema_3) & (ema_3 > ema_9)
        ema_bearish = (df['Close'] < ema_3) & (ema_3 < ema_9)

        # 3. Price action signals
        breakout_up = df['Close'] > df['High'].rolling(3).max().shift(1)  # Breaking recent high
        breakout_down = df['Close'] < df['Low'].rolling(3).min().shift(1)  # Breaking recent low

        # 4. Volume confirmation (if available)
        if 'Volume' in df.columns:
            volume_avg = df['Volume'].rolling(10).mean()
            high_volume = df['Volume'] > volume_avg * 1.2
        else:
            high_volume = pd.Series(True, index=df.index)  # Default to True if no volume

        # 5. Volatility expansion (good for scalping entries)
        vol_expansion = volatility_short > volatility_long * 1.1

        # 6. Price relative to recent range
        high_5 = df['High'].rolling(5).max()
        low_5 = df['Low'].rolling(5).min()
        range_5 = high_5 - low_5
        price_position = (df['Close'] - low_5) / range_5

        upper_range = price_position > 0.7  # In upper 30% of recent range
        lower_range = price_position < 0.3  # In lower 30% of recent range

        # Convert all conditions to boolean Series explicitly
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

        # Create boolean masks for momentum conditions
        momentum_2_bullish = (price_momentum_2 > buy_threshold).fillna(False).astype(bool)
        momentum_2_bearish = (price_momentum_2 < sell_threshold).fillna(False).astype(bool)
        price_above_sma = (df['Close'] > sma_5).fillna(False).astype(bool)
        price_below_sma = (df['Close'] < sma_5).fillna(False).astype(bool)

        # SCALPING BUY SIGNALS (More aggressive)
        scalp_buy_signals = (
            (micro_up & ema_bullish & high_volume) |  # Strong micro momentum with trend
            (breakout_up & vol_expansion) |           # Breakout with volume
            (momentum_2_bullish & ema_bullish) |      # 2-candle momentum
            (lower_range & micro_up & price_above_sma)    # Bounce from low with trend
        )

        # SCALPING SELL SIGNALS (More aggressive)
        scalp_sell_signals = (
            (micro_down & ema_bearish & high_volume) |  # Strong micro momentum against trend
            (breakout_down & vol_expansion) |           # Breakdown with volume
            (momentum_2_bearish & ema_bearish) |        # 2-candle momentum down
            (upper_range & micro_down & price_below_sma)  # Rejection from high against trend
        )

        # Additional scalping filters to reduce whipsaws
        # Avoid trading in very low volatility (sideways market)
        volatility_quantile = volatility_short.rolling(50).quantile(0.3)
        sufficient_volatility = (volatility_short > volatility_quantile).fillna(False).astype(bool)

        # Apply volatility filter
        scalp_buy_signals = scalp_buy_signals & sufficient_volatility
        scalp_sell_signals = scalp_sell_signals & sufficient_volatility

        # Ensure all signals are boolean and handle NaN values
        scalp_buy_signals = scalp_buy_signals.fillna(False).astype(bool)
        scalp_sell_signals = scalp_sell_signals.fillna(False).astype(bool)

        # Ensure no conflicting signals
        conflicting = scalp_buy_signals & scalp_sell_signals
        scalp_buy_signals = scalp_buy_signals & ~conflicting
        scalp_sell_signals = scalp_sell_signals & ~conflicting

        # Create final scalping signals with reduced hold periods
        signals = np.where(scalp_buy_signals, 2, 
                          np.where(scalp_sell_signals, 0, 1))  # 2=Buy, 1=Hold, 0=Sell

        targets['trading_signal'] = pd.Series(signals, index=df.index)

        # Debug information for scalping trading signals
        signal_counts = pd.Series(signals).value_counts()
        total_signals = len(signals)
        buy_pct = (signal_counts.get(2, 0) / total_signals) * 100
        sell_pct = (signal_counts.get(0, 0) / total_signals) * 100
        hold_pct = (signal_counts.get(1, 0) / total_signals) * 100

        print(f"SCALPING Trading Signal Distribution:")
        print(f"  Buy: {signal_counts.get(2, 0)} ({buy_pct:.1f}%)")
        print(f"  Hold: {signal_counts.get(1, 0)} ({hold_pct:.1f}%)")  
        print(f"  Sell: {signal_counts.get(0, 0)} ({sell_pct:.1f}%)")
        print(f"Base threshold: {base_threshold:.4f}, Avg dynamic threshold: {dynamic_threshold.mean():.4f}")
        print(f"Volatility range: {volatility_short.min():.4f} to {volatility_short.max():.4f}")

        # Debug information for profit_prob
        if 'profit_prob' in targets:
            profit_prob_stats = targets['profit_prob'].value_counts()
            print(f"Profit Probability Target Distribution: {profit_prob_stats.to_dict()}")
            print(f"Profit threshold used: {profit_threshold:.4f}")
            print(f"Max future return range: {max_future_return.min():.4f} to {max_future_return.max():.4f}")

        return targets

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> Dict[str, Any]:
        """Train XGBoost model for specific task."""

        # Remove NaN values and ensure we have valid targets
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        # Additional validation for target values
        if task_type == 'classification':
            # Remove any invalid target values
            valid_targets = ~np.isinf(y_clean) & (y_clean >= 0)
            X_clean = X_clean[valid_targets]
            y_clean = y_clean[valid_targets]

            # Ensure we have at least 2 classes
            unique_targets = y_clean.unique()
            if len(unique_targets) < 2:
                raise ValueError(f"Insufficient target classes for {model_name}. Found classes: {unique_targets}")
        else:
            # For regression tasks (like volatility), remove NaN and infinite values
            valid_targets = np.isfinite(y_clean) & (y_clean > 0)
            X_clean = X_clean[valid_targets]
            y_clean = y_clean[valid_targets]

            if len(y_clean) == 0:
                raise ValueError(f"No valid target values for {model_name} after cleaning")

        if len(X_clean) < 100:
            raise ValueError(f"Insufficient data for training {model_name}. Need at least 100 samples, got {len(X_clean)}")

        # Split data using time series split to avoid look-ahead bias
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X_clean))
        train_idx, test_idx = splits[-1]  # Use the last split

        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        # Scale features for regression tasks
        if task_type == 'regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Set up XGBoost parameters
        if task_type == 'classification':
            if len(np.unique(y_train)) > 2:
                objective = 'multi:softprob'
                num_class = len(np.unique(y_train))
            else:
                objective = 'binary:logistic'
                num_class = None
        else:
            objective = 'reg:squarederror'
            num_class = None

        params = {
            'objective': objective,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        if num_class:
            params['num_class'] = num_class

        # Train model
        if task_type == 'classification':
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)

        model.fit(X_train_scaled, y_train)

        # Make predictions
        if task_type == 'classification':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = None

        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }

        # Store model
        self.models[model_name] = {
            'model': model,
            'metrics': metrics,
            'feature_importance': dict(zip(self.feature_names, model.feature_importances_)),
            'task_type': task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': test_idx
        }

        return self.models[model_name]

    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all trading models."""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Prepare features
        status_text.text("Preparing features...")
        X = self.prepare_features(df)

        # Create targets
        status_text.text("Creating target variables...")
        targets = self.create_targets(df)

        models_config = [
            ('direction', 'classification'),
            ('magnitude', 'regression'),
            ('profit_prob', 'classification'),
            ('volatility', 'regression'),
            ('trend_sideways', 'classification'),
            ('reversal', 'classification'),
            ('trading_signal', 'classification')
        ]

        results = {}
        total_models = len(models_config)

        for i, (model_name, task_type) in enumerate(models_config):
            status_text.text(f"Training {model_name} model...")

            try:
                if model_name in targets:
                    result = self.train_model(model_name, X, targets[model_name], task_type)
                    results[model_name] = result
                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Target {model_name} not found")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = None

            progress_bar.progress((i + 1) / total_models)

        status_text.text("All models trained!")
        return results

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model_info = self.models[model_name]
        model = model_info['model']

        # Prepare features
        X_features = X[self.feature_names]

        # Scale if needed
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X_features)
        else:
            X_scaled = X_features.values

        # Make predictions
        predictions = model.predict(X_scaled)

        # Get probabilities for classification tasks
        if model_info['task_type'] == 'classification':
            probabilities = model.predict_proba(X_scaled)
        else:
            probabilities = None

        return predictions, probabilities

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            return {}

        return self.models[model_name]['feature_importance']