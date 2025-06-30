

import pandas as pd
import numpy as np
from typing import Dict, List

class ProfitProbabilityTechnicalIndicators:
    """Calculate technical indicators specifically for profit probability prediction."""

    @staticmethod
    def calculate_profit_probability_indicators(df):
        """Calculate indicators specifically for profit probability model"""
        df = df.copy()

        # Ensure we have the right column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        volume_col = 'Volume' if 'Volume' in df.columns else 'volume'

        try:
            # EMA calculations - 5, 10, 20
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

            # MACD histogram calculation
            ema_12 = df[close_col].ewm(span=12).mean()
            ema_26 = df[close_col].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            df['macd_histogram'] = macd - macd_signal

            # ATR calculation
            tr1 = df[high_col] - df[low_col]
            tr2 = abs(df[high_col] - df[close_col].shift(1))
            tr3 = abs(df[low_col] - df[close_col].shift(1))
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df[close_col] - bb_lower) / (bb_upper - bb_lower)

            # Donchian Channel
            df['donchian_high_20'] = df[high_col].rolling(window=20).max()
            df['donchian_low_20'] = df[low_col].rolling(window=20).min()

            # ADX calculation
            # Calculate directional movement
            plus_dm = df[high_col].diff()
            minus_dm = df[low_col].diff()
            plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
            minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
            
            # Smooth the directional movements
            plus_dm_smooth = pd.Series(plus_dm, index=df.index).rolling(window=14).mean()
            minus_dm_smooth = pd.Series(minus_dm, index=df.index).rolling(window=14).mean()
            
            # Calculate directional indicators
            plus_di = 100 * (plus_dm_smooth / df['atr'])
            minus_di = 100 * (minus_dm_smooth / df['atr'])
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()

            # Replace inf and nan values
            numeric_cols = ['ema_5', 'ema_10', 'ema_20', 'rsi_14', 'macd_histogram', 
                           'atr', 'bb_width', 'bb_position', 'donchian_high_20', 
                           'donchian_low_20', 'adx']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
        except Exception as e:
            print(f"Error calculating profit probability technical indicators: {e}")
            # Fallback calculations
            df['ema_5'] = df[close_col].ewm(span=5).mean()
            df['ema_10'] = df[close_col].ewm(span=10).mean()
            df['ema_20'] = df[close_col].ewm(span=20).mean()
            df['rsi_14'] = 50.0
            df['macd_histogram'] = 0.0
            df['atr'] = (df[high_col] - df[low_col]).rolling(14).mean()
            df['bb_width'] = df[close_col].rolling(20).std() / df[close_col].rolling(20).mean()
            df['bb_position'] = 0.5
            df['donchian_high_20'] = df[high_col].rolling(20).max()
            df['donchian_low_20'] = df[low_col].rolling(20).min()
            df['adx'] = 25.0

        return df

    @staticmethod
    def calculate_all_profit_probability_indicators(df):
        """Calculate all profit probability technical indicators"""
        print("🔧 Calculating profit probability technical indicators...")

        # Validate input data
        from utils.data_processing import DataProcessor
        is_valid, message = DataProcessor.validate_ohlc_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLC data provided: {message}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate profit probability indicators
        print("Step 1: Calculating basic profit probability indicators...")
        result_df = ProfitProbabilityTechnicalIndicators.calculate_profit_probability_indicators(result_df)
        basic_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"After basic indicators: {len(basic_features)} features")

        # Add custom engineered features for profit probability
        print("Step 2: Adding custom profit probability features...")
        from features.profit_probability_custom_engineered import add_custom_profit_probability_features
        try:
            result_df = add_custom_profit_probability_features(result_df)
            custom_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After custom features: {len(custom_features)} features")
        except Exception as e:
            print(f"Error in custom features: {e}")

        # Add lagged features for profit probability
        print("Step 3: Adding lagged profit probability features...")
        from features.profit_probability_lagged_features import add_lagged_profit_probability_features
        try:
            result_df = add_lagged_profit_probability_features(result_df)
            lagged_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After lagged features: {len(lagged_features)} features")
        except Exception as e:
            print(f"Error in lagged features: {e}")

        # Add time context features for profit probability
        print("Step 4: Adding time context features...")
        from features.profit_probability_time_context import add_profit_probability_time_context_features
        try:
            result_df = add_profit_probability_time_context_features(result_df)
            time_features = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"After time features: {len(time_features)} features")
        except Exception as e:
            print(f"Error in time features: {e}")

        # Final cleanup
        print("Step 5: Final cleanup...")
        print("Filling NaN values...")
        
        # Fill NaN values with appropriate defaults
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                result_df[col] = result_df[col].fillna(result_df[col].median())
        
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Count completely empty rows
        non_ohlc_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        empty_rows = result_df[non_ohlc_cols].isna().all(axis=1).sum()
        
        # Drop completely empty rows
        result_df = result_df.dropna(subset=non_ohlc_cols, how='all')
        
        print(f"Data points after cleanup: {len(result_df)} (dropped {empty_rows} completely empty rows)")

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Final result: {len(feature_cols)} profit probability indicators")
        print(f"Profit probability features: {feature_cols}")

        return result_df

