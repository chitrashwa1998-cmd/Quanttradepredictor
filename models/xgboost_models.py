import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingClassifier, VotingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Dict, Tuple, Any
import streamlit as st
from datetime import datetime

class QuantTradingModels:
    """Ensemble models using XGBoost, CatBoost, and Random Forest for quantitative trading predictions."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self._load_existing_models()

    def _load_existing_models(self):
        """Load previously trained models from database if available."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()
            loaded_models = db.load_trained_models()

            if loaded_models:
                # Extract feature names and ensure task_type is present
                for model_name, model_data in loaded_models.items():
                    # Ensure task_type is present, infer if missing
                    if 'task_type' not in model_data:
                        if model_name in ['direction', 'profit_prob', 'trend_sideways', 'reversal', 'trading_signal']:
                            model_data['task_type'] = 'classification'
                        elif model_name in ['magnitude', 'volatility']:
                            model_data['task_type'] = 'regression'
                        else:
                            model_data['task_type'] = 'classification'  # Default

                        # Update the loaded model with the inferred task_type
                        loaded_models[model_name] = model_data

                self.models = loaded_models
                print(f"Loaded {len(loaded_models)} existing trained models from database")

                # Extract feature names from first available model - PRIORITY ORDER
                feature_names_found = False
                for model_name, model_data in loaded_models.items():
                    if 'feature_names' in model_data and model_data['feature_names']:
                        self.feature_names = model_data['feature_names']
                        feature_names_found = True
                        print(f"Feature names loaded from {model_name}: {len(self.feature_names)} features")
                        break

                if not feature_names_found:
                    print("Warning: No feature names found in loaded models")
            else:
                print("No existing models found in database")

        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def _save_models_to_database(self):
        """Save trained models to database for persistence."""
        try:
            from utils.database_adapter import get_trading_database
            db = get_trading_database()

            # Prepare models for saving
            models_to_save = {}
            for model_name, model_data in self.models.items():
                if 'ensemble' in model_data:
                    models_to_save[model_name] = {
                        'ensemble': model_data['ensemble'],
                        'feature_names': self.feature_names,
                        'task_type': model_data.get('task_type', 'classification')
                    }

            if models_to_save:
                success = db.save_trained_models(models_to_save)
                if success:
                    print(f"Saved {len(models_to_save)} trained models to database with feature names")
                    print(f"Feature names saved: {len(self.feature_names)} features")
                else:
                    print("Failed to save models to database")

        except Exception as e:
            print(f"Error saving models to database: {str(e)}")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Remove any rows with NaN values
        df_clean = df.dropna()

        if df_clean.empty:
            raise ValueError("DataFrame is empty after removing NaN values")

        # Select feature columns (exclude OHLC and target columns)
        feature_cols = [col for col in df_clean.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        feature_cols = [col for col in feature_cols if not col.startswith(('target_', 'future_'))]

        # Remove data leakage features (post-model outputs and labels)
        leakage_features = [
            'Prediction', 'predicted_direction', 'predictions',
            'Signal', 'Signal_Name', 'Confidence',
            'accuracy', 'precision', 'recall'
        ]
        feature_cols = [col for col in feature_cols if col not in leakage_features]

        # Ensure all new candle behavior features are included
        expected_candle_features = [
            'body_size', 'upper_wick', 'lower_wick', 'total_range', 'body_ratio', 
            'wick_ratio', 'is_bullish', 'candle_strength', 'doji', 'marubozu', 
            'hammer', 'shooting_star', 'engulfing_bull', 'engulfing_bear',
            'bull_streak_3', 'bear_streak_2', 'inside_bar', 'outside_bar', 
            'reversal_bar', 'gap_up', 'gap_down', 'direction_change', 
            'momentum_surge', 'minute_of_hour', 'is_opening_range', 'is_closing_phase'
        ]

        # Add any missing candle features that exist in the dataframe (but exclude leakage features)
        for feature in expected_candle_features:
            if feature in df_clean.columns and feature not in feature_cols and feature not in leakage_features:
                feature_cols.append(feature)

        if not feature_cols:
            raise ValueError("No feature columns found. Make sure technical indicators are calculated.")

        result_df = df_clean[feature_cols]

        if result_df.empty:
            raise ValueError("Feature DataFrame is empty after column selection")

        # Store feature names for later use
        self.feature_names = list(result_df.columns)
        print(f"Feature names stored: {len(self.feature_names)} features")

        return result_df

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

        # More realistic profit threshold for 5-min scalping
        # Use smaller threshold to capture more profit opportunities
        base_profit_threshold = 0.001  # 0.1% minimum profit target (more realistic for 5-min)

        # Look ahead only 5 candles (25 minutes for 5-min data)
        future_returns_list = []
        for i in range(5):
            future_return = df['Close'].shift(-i-1) / df['Close'] - 1
            future_returns_list.append(future_return)

        # Get maximum return within 5 periods
        future_returns_df = pd.concat(future_returns_list, axis=1)
        max_future_return = future_returns_df.max(axis=1)

        # Use adaptive threshold based on actual data distribution
        # Aim for 30-40% profit opportunities (more balanced)
        profit_threshold = np.percentile(max_future_return.dropna(), 65)  # Top 35% as profit opportunities
        profit_threshold = max(profit_threshold, base_profit_threshold)  # Ensure minimum threshold

        targets['profit_prob'] = (max_future_return > profit_threshold).astype(int)

        # 4. Volatility forecasting (next period volatility)
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
            print(f"  25th percentile: {vol_stats['25%']:.6f}")
            print(f"  75th percentile: {vol_stats['75%']:.6f}")

        targets['volatility'] = future_vol

        # 5. Trend vs sideways classification - IMPROVED ALGORITHM
        # Use historical data only (no look-ahead bias)

        # Calculate moving averages for trend detection
        sma_5 = df['Close'].rolling(5).mean()
        sma_10 = df['Close'].rolling(10).mean()
        sma_20 = df['Close'].rolling(20).mean()
        ema_8 = df['Close'].ewm(span=8).mean()
        ema_21 = df['Close'].ewm(span=21).mean()

        # Calculate price momentum (historical only)
        price_change_5 = df['Close'] / df['Close'].shift(5) - 1  # 5-period momentum
        price_change_10 = df['Close'] / df['Close'].shift(10) - 1  # 10-period momentum
        price_change_20 = df['Close'] / df['Close'].shift(20) - 1  # 20-period momentum

        # Calculate volatility for adaptive thresholds
        returns = df['Close'].pct_change()
        volatility_10 = returns.rolling(10).std()
        volatility_20 = returns.rolling(20).std()

        # Data-adaptive trend threshold based on actual volatility distribution
        # Calculate percentiles of actual data volatility
        vol_25th = volatility_20.quantile(0.25)
        vol_50th = volatility_20.quantile(0.50)
        vol_75th = volatility_20.quantile(0.75)

        # Use adaptive base threshold based on data characteristics
        base_threshold = np.maximum(0.001, vol_50th * 0.8)  # 80% of median volatility
        volatility_multiplier = 1.0
        trend_threshold = np.maximum(base_threshold, volatility_20 * volatility_multiplier)

        # 1. MOVING AVERAGE TREND STRENGTH
        # EMA alignment (strong trend indicator)
        ema_bullish_trend = (ema_8 > ema_21) & (df['Close'] > ema_8)
        ema_bearish_trend = (ema_8 < ema_21) & (df['Close'] < ema_8)
        ema_trend_strength = ema_bullish_trend | ema_bearish_trend

        # SMA slope analysis
        sma_20_slope = (sma_20 - sma_20.shift(5)) / sma_20.shift(5)
        strong_sma_trend = np.abs(sma_20_slope) > (trend_threshold * 0.5)

        # 2. MOMENTUM ANALYSIS
        # Consistent momentum across multiple timeframes
        momentum_5_strong = np.abs(price_change_5) > trend_threshold
        momentum_10_strong = np.abs(price_change_10) > (trend_threshold * 1.2)
        momentum_20_strong = np.abs(price_change_20) > (trend_threshold * 1.5)

        # Momentum consistency (same direction across timeframes)
        momentum_consistent = (
            (price_change_5 > 0) & (price_change_10 > 0) & (price_change_20 > 0)
        ) | (
            (price_change_5 < 0) & (price_change_10 < 0) & (price_change_20 < 0)
        )

        # 3. VOLATILITY REGIME
        # Higher volatility often indicates trending market
        volatility_expansion = volatility_10 > (volatility_20 * 1.2)

        # 4. PRICE POSITION RELATIVE TO MOVING AVERAGES
        # Price consistently above/below key moving averages
        price_vs_sma20 = df['Close'] / sma_20 - 1
        strong_price_position = np.abs(price_vs_sma20) > (trend_threshold * 0.5)

        # 5. TREND PERSISTENCE
        # Check if trend conditions have persisted
        trend_persistence_3 = (
            ema_trend_strength & 
            ema_trend_strength.shift(1) & 
            ema_trend_strength.shift(2)
        )

        # MULTI-REGIME TREND CLASSIFICATION
        # Create three trend strength levels for better market regime detection

        # 1. STRONG TRENDS (high conviction)
        strong_trend_strict = (
            momentum_consistent & 
            (momentum_10_strong | momentum_20_strong) &
            ema_trend_strength &
            strong_price_position
        )

        # 2. MODERATE TRENDS (medium conviction)
        moderate_trend = (
            ((momentum_10_strong | momentum_20_strong) & ema_trend_strength) |
            (momentum_consistent & strong_sma_trend) |
            (volatility_expansion & strong_price_position & ema_trend_strength)
        ) & ~strong_trend_strict

        # 3. WEAK TRENDS (low conviction but still directional)
        weak_trend = (
            (momentum_5_strong & ema_trend_strength) |
            (trend_persistence_3 & (strong_price_position | strong_sma_trend)) |
            (volatility_expansion & (momentum_5_strong | strong_sma_trend))
        ) & ~strong_trend_strict & ~moderate_trend

        # FINAL BINARY CLASSIFICATION with adaptive thresholds
        # Use percentile-based approach to maintain reasonable balance
        all_trend_strength = strong_trend_strict.astype(int) * 3 + moderate_trend.astype(int) * 2 + weak_trend.astype(int) * 1

        # Adaptive threshold: aim for 10-20% trending periods (realistic for 5-min data)
        trend_threshold_percentile = 85  # Top 15% as trending
        trend_cutoff = np.percentile(all_trend_strength, trend_threshold_percentile)

        strong_trend = all_trend_strength >= trend_cutoff

        # Convert to binary: 1 = trending, 0 = sideways
        targets['trend_sideways'] = strong_trend.astype(int)

        # Debug information for trend_sideways
        trend_counts = targets['trend_sideways'].value_counts()
        print(f"Trend/Sideways Distribution: Trending={trend_counts.get(1, 0)}, Sideways={trend_counts.get(0, 0)}")
        if len(trend_counts) > 0:
            print(f"Trending percentage: {trend_counts.get(1, 0) / len(targets['trend_sideways']) * 100:.1f}%")

        # 6. Reversal points - PRACTICAL DETECTION (NO LOOK-AHEAD BIAS)
        # Calculate momentum and trend indicators for reversal detection
        price_change_1 = df['Close'].pct_change(1)
        price_change_3 = df['Close'].pct_change(3)
        price_change_5 = df['Close'].pct_change(5)

        # Calculate RSI-like momentum indicator (historical data only)
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
        volatility = df['Close'].pct_change().rolling(10).std()
        base_volatility_threshold = 0.005  # 0.5% base threshold

        # HISTORICAL-ONLY REVERSAL CONDITIONS (NO FUTURE DATA)

        # BULLISH REVERSAL CONDITIONS (Price expected to bounce up)
        # 1. Price at or near recent lows with oversold conditions
        near_lows = price_position <= 0.25  # Bottom 25% of recent range
        oversold_momentum = momentum_index <= 35  # Oversold RSI
        recent_decline = price_change_3 < -0.003  # Recent decline of 0.3%+

        # 2. Price below key moving averages (potential support)
        below_sma_short = df['Close'] < sma_short
        below_sma_medium = df['Close'] < sma_medium

        # 3. Volatility expansion (often precedes reversals)
        vol_expansion = volatility > volatility.rolling(20).mean() * 1.2

        # 4. Price action patterns (hammer-like candles)
        candle_body = np.abs(df['Close'] - df['Open'])
        candle_range = df['High'] - df['Low']
        lower_wick = df['Open'].combine(df['Close'], min) - df['Low']
        upper_wick = df['High'] - df['Open'].combine(df['Close'], max)

        # Hammer pattern: small body, long lower wick, short upper wick
        hammer_pattern = (
            (lower_wick > candle_body * 2) &  # Lower wick > 2x body
            (upper_wick < candle_body * 0.5) &  # Upper wick < 0.5x body
            (candle_range > 0)  # Valid candle
        )

        # BEARISH REVERSAL CONDITIONS (Price expected to decline)
        # 1. Price at or near recent highs with overbought conditions
        near_highs = price_position >= 0.75  # Top 25% of recent range
        overbought_momentum = momentum_index >= 65  # Overbought RSI
        recent_rally = price_change_3 > 0.003  # Recent rally of 0.3%+

        # 2. Price above key moving averages (potential resistance)
        above_sma_short = df['Close'] > sma_short
        above_sma_medium = df['Close'] > sma_medium

        # 3. Shooting star pattern: small body, long upper wick, short lower wick
        shooting_star_pattern = (
            (upper_wick > candle_body * 2) &  # Upper wick > 2x body
            (lower_wick < candle_body * 0.5) &  # Lower wick < 0.5x body
            (candle_range > 0)  # Valid candle
        )

        # BULLISH REVERSAL SIGNAL (Multiple scenarios)
        bullish_reversal_strict = (
            near_lows & oversold_momentum & recent_decline  # Classic oversold bounce
        )

        bullish_reversal_moderate = (
            (near_lows & (oversold_momentum | recent_decline)) |  # Partial oversold
            (below_sma_short & oversold_momentum & vol_expansion) |  # Support with momentum
            (hammer_pattern & below_sma_medium & recent_decline)  # Hammer at support
        )

        # BEARISH REVERSAL SIGNAL (Multiple scenarios)
        bearish_reversal_strict = (
            near_highs & overbought_momentum & recent_rally  # Classic overbought decline
        )

        bearish_reversal_moderate = (
            (near_highs & (overbought_momentum | recent_rally)) |  # Partial overbought
            (above_sma_short & overbought_momentum & vol_expansion) |  # Resistance with momentum
            (shooting_star_pattern & above_sma_medium & recent_rally)  # Shooting star at resistance
        )

        # COMBINE ALL REVERSAL SIGNALS
        # Use a balanced approach - not too strict, not too loose
        bullish_reversal = bullish_reversal_strict | bullish_reversal_moderate
        bearish_reversal = bearish_reversal_strict | bearish_reversal_moderate

        # Ensure we don't have conflicting signals
        conflicting_reversals = bullish_reversal & bearish_reversal
        bullish_reversal = bullish_reversal & ~conflicting_reversals
        bearish_reversal = bearish_reversal & ~conflicting_reversals

        # Final reversal signal: 1 = reversal expected, 0 = no reversal
        reversal_signal = (bullish_reversal | bearish_reversal).astype(int)

        # Apply minimum data filter (need enough history for indicators)
        reversal_signal.iloc[:momentum_window] = 0  # First 14 periods = no signal

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

        # SCALPING THRESHOLDS - More aggressive for balanced signals
        base_threshold = 0.0008  # 0.08% base threshold (more sensitive)
        volatility_multiplier = 0.2  # Even lower multiplier for more signals

        # Dynamic thresholds based on recent volatility
        dynamic_threshold = np.maximum(base_threshold, volatility_short * volatility_multiplier)
        buy_threshold = dynamic_threshold * 0.7  # Make buy signals easier
        sell_threshold = -dynamic_threshold * 0.7  # Make sell signals easier

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

        # SCALPING BUY SIGNALS (Much more aggressive)
        scalp_buy_signals = (
            (micro_up & ema_bullish) |                    # Micro momentum with trend (no volume requirement)
            (breakout_up) |                               # Any breakout (no volume requirement)
            (momentum_2_bullish) |                        # 2-candle momentum (standalone)
            (lower_range & micro_up) |                    # Bounce from low (no SMA requirement)
            (ema_bullish & price_above_sma) |             # Trend alignment
            (micro_up & vol_expansion)                    # Momentum with volatility
        )

        # SCALPING SELL SIGNALS (Much more aggressive)
        scalp_sell_signals = (
            (micro_down & ema_bearish) |                  # Micro momentum against trend (no volume requirement)
            (breakout_down) |                             # Any breakdown (no volume requirement)
            (momentum_2_bearish) |                        # 2-candle momentum down (standalone)
            (upper_range & micro_down) |                  # Rejection from high (no SMA requirement)
            (ema_bearish & price_below_sma) |             # Trend alignment
            (micro_down & vol_expansion)                  # Momentum with volatility
        )

        # Additional scalping filters to reduce whipsaws (more permissive)
        # Only avoid extremely low volatility periods
        volatility_quantile = volatility_short.rolling(50).quantile(0.1)  # Only bottom 10%
        sufficient_volatility = (volatility_short > volatility_quantile).fillna(True).astype(bool)  # Default to True

        # Apply volatility filter (more permissive)
        scalp_buy_signals = scalp_buy_signals & sufficient_volatility
        scalp_sell_signals = scalp_sell_signals & sufficient_volatility

        # Ensure all signals are boolean and handle NaN values
        scalp_buy_signals = scalp_buy_signals.fillna(False).astype(bool)
        scalp_sell_signals = scalp_sell_signals.fillna(False).astype(bool)

        # Ensure no conflicting signals
        conflicting = scalp_buy_signals & scalp_sell_signals
        scalp_buy_signals = scalp_buy_signals & ~conflicting
        scalp_sell_signals = scalp_sell_signals & ~conflicting

        # Create final scalping signals with balanced distribution
        signals = np.where(scalp_buy_signals, 2, 
                          np.where(scalp_sell_signals, 0, 1))  # 2=Buy, 1=Hold, 0=Sell

        # Force better distribution if too many holds
        signal_counts = pd.Series(signals).value_counts()
        hold_percentage = signal_counts.get(1, 0) / len(signals) * 100

        if hold_percentage > 80:  # If more than 80% holds, be more aggressive
            # Convert some holds to signals based on price momentum
            price_change_small = df['Close'].pct_change(1).fillna(0)

            # Convert holds with positive momentum to buys
            additional_buys = (signals == 1) & (price_change_small > 0.0005)
            # Convert holds with negative momentum to sells
            additional_sells = (signals == 1) & (price_change_small < -0.0005)

            signals = np.where(additional_buys, 2, signals)
            signals = np.where(additional_sells, 0, signals)

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

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification', train_split: float = 0.8) -> Dict[str, Any]:
        """Train ensemble model using multiple algorithms with voting."""

        # Ensure X and y have the same index for proper alignment
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        # Remove NaN values and ensure we have valid targets
        mask = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
        X_clean = X_aligned[mask]
        y_clean = y_aligned[mask]

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

        # Use configurable split with time-based ordering (no shuffling for time series data)
        split_idx = int(len(X_clean) * train_split)

        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        y_train = y_clean.iloc[:split_idx]
        y_test = y_clean.iloc[split_idx:]

        print(f"Training on {len(X_train)} samples ({len(X_train)/len(X_clean)*100:.1f}%), testing on {len(X_test)} samples ({len(X_test)/len(X_clean)*100:.1f}%)")

        # Scale features for all models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[model_name] = scaler

        # Define base model parameters
        random_state = 42

        if task_type == 'classification':
            # Classification ensemble: XGBoost + CatBoost + Random Forest

            # XGBoost Classifier
            xgb_model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )

            # CatBoost Classifier
            catboost_model = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False
            )

            # Random Forest Classifier
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
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
                voting='soft'
            )
            
            # Apply calibration for direction model to address overconfidence
            if model_name == 'direction':
                print(f"Applying calibration to {model_name} model to reduce overconfidence...")
                # Use cross-validation for calibration (more robust than prefit)
                ensemble_model = CalibratedClassifierCV(
                    base_ensemble, 
                    method="sigmoid",  # Platt scaling
                    cv=3  # 3-fold cross-validation for calibration
                )
            else:
                ensemble_model = base_ensemble

        else:
            # Regression ensemble: XGBoost + CatBoost + Random Forest

            # XGBoost Regressor
            xgb_model = xgb.XGBRegressor(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )

            # CatBoost Regressor
            catboost_model = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False
            )

            # Random Forest Regressor
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

        # Train ensemble model with error handling
        try:
            ensemble_model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"Error during ensemble training for {model_name}: {str(e)}")
            # If calibration fails, fall back to base ensemble
            if model_name == 'direction':
                print(f"Calibration failed for {model_name}, falling back to uncalibrated ensemble...")
                ensemble_model = base_ensemble
                ensemble_model.fit(X_train_scaled, y_train)
            else:
                raise e

        # Make predictions
        if task_type == 'classification':
            y_pred = ensemble_model.predict(X_test_scaled)
            y_pred_proba = ensemble_model.predict_proba(X_test_scaled)
        else:
            y_pred = ensemble_model.predict(X_test_scaled)
            y_pred_proba = None

        # Calculate metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Add calibration information for direction model
            if model_name == 'direction':
                try:
                    # Calculate calibration metrics
                    from sklearn.calibration import calibration_curve
                    
                    # Get calibrated probabilities
                    prob_pos = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate calibration curve (reliability diagram)
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_test, prob_pos, n_bins=10
                    )
                    
                    # Calculate Brier score (lower is better)
                    brier_score = np.mean((prob_pos - y_test) ** 2)
                    
                    # Check if model was successfully calibrated
                    is_calibrated = hasattr(ensemble_model, 'calibrated_classifiers_')
                    
                    metrics['calibration'] = {
                        'brier_score': brier_score,
                        'is_calibrated': is_calibrated,
                        'calibration_method': 'Platt Scaling (Sigmoid)' if is_calibrated else 'None (Calibration Failed)',
                        'mean_predicted_probability': np.mean(prob_pos),
                        'actual_positive_rate': np.mean(y_test)
                    }
                    
                    print(f"Calibration results for {model_name}:")
                    print(f"  Successfully calibrated: {is_calibrated}")
                    print(f"  Brier Score: {brier_score:.4f} (lower is better)")
                    print(f"  Mean Predicted Probability: {np.mean(prob_pos):.3f}")
                    print(f"  Actual Positive Rate: {np.mean(y_test):.3f}")
                    
                except Exception as e:
                    print(f"Error calculating calibration metrics for {model_name}: {str(e)}")
                    metrics['calibration'] = {
                        'brier_score': None,
                        'is_calibrated': False,
                        'calibration_method': 'Error during calibration',
                        'error': str(e)
                    }
            
            # Calculate individual model accuracies for comparison
            individual_scores = {}
            
            # For calibrated models, get base estimators differently
            if hasattr(ensemble_model, 'calibrated_classifiers_'):
                # This is a calibrated classifier
                calibrated_classifier = ensemble_model.calibrated_classifiers_[0]
                
                # Try different attribute names for compatibility
                if hasattr(calibrated_classifier, 'estimator'):
                    base_classifier = calibrated_classifier.estimator
                elif hasattr(calibrated_classifier, 'base_estimator'):
                    base_classifier = calibrated_classifier.base_estimator
                else:
                    base_classifier = None
                
                if base_classifier and hasattr(base_classifier, 'named_estimators_'):
                    for name, model in base_classifier.named_estimators_.items():
                        individual_pred = model.predict(X_test_scaled)
                        individual_scores[f'{name}_accuracy'] = accuracy_score(y_test, individual_pred)
            elif hasattr(ensemble_model, 'named_estimators_'):
                # This is a regular ensemble
                for name, model in ensemble_model.named_estimators_.items():
                    individual_pred = model.predict(X_test_scaled)
                    individual_scores[f'{name}_accuracy'] = accuracy_score(y_test, individual_pred)

            metrics.update(individual_scores)

        else:
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
            # Handle calibrated classifiers
            if hasattr(ensemble_model, 'calibrated_classifiers_'):
                # For calibrated models, access the base estimator
                calibrated_classifier = ensemble_model.calibrated_classifiers_[0]
                
                # Try different attribute names for compatibility
                if hasattr(calibrated_classifier, 'estimator'):
                    base_classifier = calibrated_classifier.estimator
                elif hasattr(calibrated_classifier, 'base_estimator'):
                    base_classifier = calibrated_classifier.base_estimator
                else:
                    raise AttributeError("Cannot find base estimator in calibrated classifier")
                
                if hasattr(base_classifier, 'named_estimators_'):
                    xgb_estimator = base_classifier.named_estimators_['xgboost']
                else:
                    xgb_estimator = base_classifier
            else:
                xgb_estimator = ensemble_model.named_estimators_['xgboost']
            
            feature_importance = dict(zip(self.feature_names, xgb_estimator.feature_importances_))
        except Exception as e:
            print(f"Could not extract feature importance: {e}")
            feature_importance = {}

        # Store model
        self.models[model_name] = {
            'ensemble': ensemble_model,  # Changed 'model' to 'ensemble' for database compatibility
            'metrics': metrics,
            'feature_importance': feature_importance,
            'task_type': task_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_indices': X_test.index,
            'ensemble_type': 'voting_classifier' if task_type == 'classification' else 'voting_regressor',
            'base_models': list(ensemble_model.named_estimators_.keys())
        }

        return self.models[model_name]

    def train_all_models(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict[str, Any]:
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
                    # Ensure X and target are properly aligned by using common index
                    target_series = targets[model_name]
                    common_index = X.index.intersection(target_series.index)

                    if len(common_index) == 0:
                        st.warning(f"⚠️ No common indices between features and {model_name} target")
                        results[model_name] = None
                        continue

                    X_aligned = X.loc[common_index]
                    y_aligned = target_series.loc[common_index]

                    result = self.train_model(model_name, X_aligned, y_aligned, task_type, train_split)
                    results[model_name] = result
                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Target {model_name} not found")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = None

            progress_bar.progress((i + 1) / total_models)

        status_text.text("Saving trained models to database...")
        # Automatically save all trained models for persistence
        self._save_models_to_database()

        status_text.text("All models trained and saved!")
        return results

    def train_selected_models(self, df: pd.DataFrame, selected_models: list, train_split: float = 0.8) -> Dict[str, Any]:
        """Train only selected models."""
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
        total_models = len(selected_models)

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name} model...")

            try:
                # Find the task type for the current model
                task_type = None
                for m, t in models_config:
                    if m == model_name:
                        task_type = t
                        break

                if model_name in targets and task_type is not None:
                    # Ensure X and target are properly aligned by using common index
                    target_series = targets[model_name]
                    common_index = X.index.intersection(target_series.index)

                    if len(common_index) == 0:
                        st.warning(f"⚠️ No common indices between features and {model_name} target")
                        results[model_name] = None
                        continue

                    X_aligned = X.loc[common_index]
                    y_aligned = target_series.loc[common_index]

                    result = self.train_model(model_name, X_aligned, y_aligned, task_type, train_split)
                    results[model_name] = result
                    st.success(f"✅ {model_name} model trained successfully")
                else:
                    st.warning(f"⚠️ Target {model_name} not found or task type not defined")
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = None

            progress_bar.progress((i + 1) / total_models)

        status_text.text("Saving trained models to database...")
        # Automatically save all trained models for persistence
        self._save_models_to_database()

        status_text.text("Selected models trained and saved!")
        return results

    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using trained ensemble model with enhanced confidence calculation."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")

        model_info = self.models[model_name]
        # Handle both new 'ensemble' and legacy 'model' keys
        model = model_info.get('ensemble') or model_info.get('model')

        # Try to get feature names from multiple sources
        feature_names = None

        # 1. Check instance attribute
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_names = self.feature_names

        # 2. Check model info
        elif 'feature_names' in model_info and model_info['feature_names']:
            feature_names = model_info['feature_names']
            self.feature_names = feature_names  # Update instance

        # 3. Try to infer from input data (fallback)
        elif not X.empty:
            # Use all available features excluding OHLC, target columns, and leakage features
            leakage_features = [
                'Prediction', 'predicted_direction', 'predictions',
                'Signal', 'Signal_Name', 'Confidence',
                'accuracy', 'precision', 'recall'
            ]
            available_features = [col for col in X.columns 
                                if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                                and not col.startswith(('target_', 'future_'))
                                and col not in leakage_features]
            if available_features:
                feature_names = available_features
                self.feature_names = feature_names
                print(f"Warning: Inferred {len(feature_names)} feature names from input data")

        if not feature_names:
            raise ValueError(f"No feature names found for model {model_name}. Model may not be properly trained. Try retraining the model.")

        # Validate input features
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        missing_features = [col for col in feature_names if col not in X.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            # Use only available features
            available_features = [col for col in feature_names if col in X.columns]
            if not available_features:
                raise ValueError(f"No required features found in input data. Expected: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
            feature_names = available_features

        # Prepare features
        X_features = X[feature_names]

        if X_features.empty:
            raise ValueError("Feature DataFrame is empty after column selection")

        # Scale features (all ensemble models use scaling)
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X_features)
        else:
            X_scaled = X_features.values

        # Make predictions using ensemble
        predictions = model.predict(X_scaled)

        # Enhanced confidence calculation for classification tasks
        if model_info['task_type'] == 'classification':
            # Handle calibrated models vs regular ensemble models
            if hasattr(model, 'calibrated_classifiers_'):
                # This is a calibrated model - probabilities are already well-calibrated
                calibrated_probabilities = model.predict_proba(X_scaled)
                
                # For calibrated models, use the calibrated probabilities directly
                n_samples = len(predictions)
                confidence_scores = np.max(calibrated_probabilities, axis=1)
                
                # Create probability matrix
                probabilities = calibrated_probabilities
                
                print(f"Using calibrated probabilities for {model_name}")
                print(f"Confidence range: {confidence_scores.min():.3f} - {confidence_scores.max():.3f}")
                
            else:
                # Regular ensemble model - use original confidence calculation
                individual_predictions = []
                individual_probabilities = []

                # Handle both calibrated and regular ensemble models
                if hasattr(model, 'calibrated_classifiers_'):
                    # For calibrated models, get the base estimator
                    calibrated_classifier = model.calibrated_classifiers_[0]
                    
                    # Try different attribute names for compatibility
                    if hasattr(calibrated_classifier, 'estimator'):
                        base_estimator = calibrated_classifier.estimator
                    elif hasattr(calibrated_classifier, 'base_estimator'):
                        base_estimator = calibrated_classifier.base_estimator
                    else:
                        # Fallback: use the whole model as base estimator
                        base_estimator = model
                    
                    if hasattr(base_estimator, 'named_estimators_'):
                        estimators = base_estimator.named_estimators_
                    else:
                        # Single estimator case
                        estimators = {'base': base_estimator}
                else:
                    # Regular ensemble
                    estimators = model.named_estimators_

                for name, individual_model in estimators.items():
                    ind_pred = individual_model.predict(X_scaled)
                    individual_predictions.append(ind_pred)

                    # Get probabilities if available
                    if hasattr(individual_model, 'predict_proba'):
                        ind_proba = individual_model.predict_proba(X_scaled)
                        individual_probabilities.append(ind_proba)

                # Calculate confidence based on model agreement and probability strength
                n_samples = len(predictions)
                confidence_scores = np.zeros(n_samples)

                for i in range(n_samples):
                    # Method 1: Model agreement (how many models agree with final prediction)
                    individual_preds_at_i = [pred[i] for pred in individual_predictions]
                    agreement_score = sum(1 for pred in individual_preds_at_i if pred == predictions[i]) / len(individual_preds_at_i)

                    # Method 2: Average probability strength (how confident individual models are)
                    if individual_probabilities:
                        prob_strengths = []
                        for j, proba_matrix in enumerate(individual_probabilities):
                            max_prob = np.max(proba_matrix[i])  # Highest probability for this sample
                            prob_strengths.append(max_prob)
                        avg_prob_strength = np.mean(prob_strengths)
                    else:
                        avg_prob_strength = 0.5

                    # Combined confidence: weighted average of agreement and probability strength
                    confidence_scores[i] = 0.6 * agreement_score + 0.4 * avg_prob_strength

                    # Ensure minimum confidence for unanimous decisions
                    if agreement_score == 1.0:  # All models agree
                        confidence_scores[i] = max(confidence_scores[i], 0.75)

                # Create probability matrix with confidence as max probability
                probabilities = np.zeros((n_samples, 2))
                for i in range(n_samples):
                    if predictions[i] == 1:  # Predicted Up
                        probabilities[i, 1] = confidence_scores[i]
                        probabilities[i, 0] = 1 - confidence_scores[i]
                    else:  # Predicted Down
                        probabilities[i, 0] = confidence_scores[i]
                        probabilities[i, 1] = 1 - confidence_scores[i]

            
        else:
            probabilities = None

        return predictions, probabilities

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            return {}

        return self.models[model_name]['feature_importance']