
import pandas as pd
import numpy as np
from typing import Dict, List

class DirectionTechnicalIndicators:
    """Calculate direction-specific technical indicators for price movement prediction."""

    @staticmethod
    def calculate_direction_indicators(df):
        """Calculate indicators specifically for direction model"""
        df = df.copy()

        # Ensure we have the right column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

        try:
            # RSI calculation
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD calculation
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()

            # EMA fast and slow
            df['ema_fast'] = df[close_col].ewm(span=12).mean()
            df['ema_slow'] = df[close_col].ewm(span=26).mean()

            # ADX calculation (Average Directional Index)
            high_low = df[high_col] - df[low_col]
            high_close = np.abs(df[high_col] - df[close_col].shift())
            low_close = np.abs(df[low_col] - df[close_col].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            plus_dm = df[high_col].diff()
            minus_dm = df[low_col].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = np.abs(minus_dm)
            
            plus_di = 100 * (plus_dm.rolling(14).mean() / true_range.rolling(14).mean())
            minus_di = 100 * (minus_dm.rolling(14).mean() / true_range.rolling(14).mean())
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(14).mean()

            # On-Balance Volume (OBV)
            if volume_col in df.columns:
                obv = [0]
                for i in range(1, len(df)):
                    if df[close_col].iloc[i] > df[close_col].iloc[i-1]:
                        obv.append(obv[-1] + df[volume_col].iloc[i])
                    elif df[close_col].iloc[i] < df[close_col].iloc[i-1]:
                        obv.append(obv[-1] - df[volume_col].iloc[i])
                    else:
                        obv.append(obv[-1])
                df['obv'] = obv
            else:
                df['obv'] = 0

            # Stochastic Oscillator
            lowest_low = df[low_col].rolling(window=14).min()
            highest_high = df[high_col].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Williams %R
            df['williams_r'] = -100 * ((highest_high - df[close_col]) / (highest_high - lowest_low))

            # Momentum
            df['momentum'] = df[close_col] / df[close_col].shift(10) - 1

            # Rate of Change (ROC)
            df['roc'] = ((df[close_col] - df[close_col].shift(12)) / df[close_col].shift(12)) * 100

            # Replace inf and nan values
            direction_features = ['rsi', 'macd', 'macd_signal', 'ema_fast', 'ema_slow', 'adx', 
                                'obv', 'stoch_k', 'stoch_d', 'williams_r', 'momentum', 'roc']
            
            for col in direction_features:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
        except Exception as e:
            print(f"Error calculating direction indicators: {e}")
            # Fallback calculations
            df['rsi'] = 50.0  # Neutral RSI
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['ema_fast'] = df[close_col]
            df['ema_slow'] = df[close_col]
            df['adx'] = 25.0  # Neutral ADX
            df['obv'] = 0.0
            df['stoch_k'] = 50.0
            df['stoch_d'] = 50.0
            df['williams_r'] = -50.0
            df['momentum'] = 0.0
            df['roc'] = 0.0

        return df

    @staticmethod
    def calculate_all_direction_indicators(df):
        """Calculate all direction-specific technical indicators"""
        print("🔧 Calculating direction-specific technical indicators...")

        # Validate input data
        from utils.data_processing import DataProcessor
        is_valid, message = DataProcessor.validate_ohlc_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLC data provided: {message}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate direction indicators
        result_df = DirectionTechnicalIndicators.calculate_direction_indicators(result_df)

        # Add custom engineered features for direction
        from features.direction_custom_engineered import compute_direction_custom_features
        result_df = compute_direction_custom_features(result_df)

        # Add lagged features for direction
        from features.direction_lagged_features import add_direction_lagged_features
        result_df = add_direction_lagged_features(result_df)

        # Add time context features for direction
        from features.direction_time_context import add_direction_time_context_features
        result_df = add_direction_time_context_features(result_df)

        # Final cleanup
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.dropna()

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Calculated {len(feature_cols)} direction-specific indicators")
        print(f"Direction features: {feature_cols}")

        return result_df
