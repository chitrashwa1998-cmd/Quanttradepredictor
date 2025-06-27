
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
            # EMA calculations - only 5, 10, 20
            df['ema_5'] = df[close_col].ewm(span=5).mean()
            df['ema_10'] = df[close_col].ewm(span=10).mean()
            df['ema_20'] = df[close_col].ewm(span=20).mean()

            # RSI calculation (14 period)
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD calculation
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            df['macd_histogram'] = macd - macd_signal

            # Bollinger Bands (20 period, 2 std)
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            # Bollinger Band position (0 = at lower band, 1 = at upper band)
            df['bollinger_band_position'] = (df[close_col] - bb_lower) / (bb_upper - bb_lower)

            # Stochastic Oscillator (14 period)
            lowest_low = df[low_col].rolling(window=14).min()
            highest_high = df[high_col].rolling(window=14).max()
            df['stochastic_k'] = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))
            df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()

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

            # Donchian Channel (20 period)
            df['donchian_high_20'] = df[high_col].rolling(window=20).max()
            df['donchian_low_20'] = df[low_col].rolling(window=20).min()

            # Replace inf and nan values
            direction_features = ['ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd_histogram', 
                                'bollinger_band_position', 'bb_width', 'stochastic_k', 'stochastic_d', 
                                'adx', 'obv', 'donchian_high_20', 'donchian_low_20']
            
            for col in direction_features:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
        except Exception as e:
            print(f"Error calculating direction indicators: {e}")
            # Fallback calculations
            df['ema_5'] = df[close_col]
            df['ema_10'] = df[close_col]
            df['ema_20'] = df[close_col]
            df['rsi_14'] = 50.0  # Neutral RSI
            df['macd_histogram'] = 0.0
            df['bollinger_band_position'] = 0.5  # Middle position
            df['bb_width'] = 0.1
            df['stochastic_k'] = 50.0
            df['stochastic_d'] = 50.0
            df['adx'] = 25.0  # Neutral ADX
            df['obv'] = 0.0
            df['donchian_high_20'] = df[high_col]
            df['donchian_low_20'] = df[low_col]

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
        from features.direction_custom_engineered import add_custom_direction_features
        result_df = add_custom_direction_features(result_df)

        # Add lagged features for direction
        from features.direction_lagged_features import add_lagged_direction_features
        result_df = add_lagged_direction_features(result_df)

        # Add time context features for direction
        from features.direction_time_context import add_time_context_features
        result_df = add_time_context_features(result_df)

        # Final cleanup
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.dropna()

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Calculated {len(feature_cols)} direction-specific indicators")
        print(f"Direction features: {feature_cols}")

        return result_df
