import pandas as pd
import numpy as np
from typing import Dict, List

class TechnicalIndicators:
    """Calculate volatility-specific technical indicators for trading analysis."""

    @staticmethod
    def calculate_volatility_indicators(df):
        """Calculate indicators specifically for volatility model"""
        df = df.copy()

        # Ensure we have the right column names
        close_col = 'Close' if 'Close' in df.columns else 'close'
        open_col = 'Open' if 'Open' in df.columns else 'open'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'

        try:
            # Calculate True Range for ATR
            tr1 = df[high_col] - df[low_col]
            tr2 = abs(df[high_col] - df[close_col].shift(1))
            tr3 = abs(df[low_col] - df[close_col].shift(1))
            true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()

            # Bollinger Bands calculation
            bb_period = 20
            bb_std = 2
            bb_middle = df[close_col].rolling(bb_period).mean()
            bb_std_dev = df[close_col].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # Keltner Channel calculation
            kc_period = 20
            kc_multiplier = 2
            kc_middle = df[close_col].rolling(kc_period).mean()
            kc_atr = true_range.rolling(kc_period).mean()
            kc_upper = kc_middle + (kc_atr * kc_multiplier)
            kc_lower = kc_middle - (kc_atr * kc_multiplier)
            df['keltner_width'] = (kc_upper - kc_lower) / kc_middle

            # RSI calculation
            delta = df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Donchian Channel calculation
            dc_period = 20
            dc_upper = df[high_col].rolling(dc_period).max()
            dc_lower = df[low_col].rolling(dc_period).min()
            df['donchian_width'] = (dc_upper - dc_lower) / df[close_col]
            
            # Replace inf and nan values
            for col in ['atr', 'bb_width', 'keltner_width', 'rsi', 'donchian_width']:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            # Fallback calculations to ensure we have all required features
            df['atr'] = (df[high_col] - df[low_col]).rolling(14).mean()
            df['bb_width'] = df[close_col].rolling(20).std() / df[close_col].rolling(20).mean()
            df['keltner_width'] = df[close_col].rolling(20).std() / df[close_col].rolling(20).mean()
            df['rsi'] = 50.0  # Neutral RSI
            df['donchian_width'] = (df[high_col].rolling(20).max() - df[low_col].rolling(20).min()) / df[close_col]

        return df

    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators for the dataset"""
        print("🔧 Calculating comprehensive technical indicators...")

        # Validate input data
        from utils.data_processing import DataProcessor
        is_valid, message = DataProcessor.validate_ohlc_data(df)
        if not is_valid:
            raise ValueError(f"Invalid OHLC data provided: {message}")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate basic indicators
        result_df = TechnicalIndicators.calculate_volatility_indicators(result_df)

        # Add custom engineered features
        from features.custom_engineered import compute_custom_volatility_features
        result_df = compute_custom_volatility_features(result_df)

        # Add lagged features
        from features.lagged_features import add_volatility_lagged_features
        result_df = add_volatility_lagged_features(result_df)

        # Add time context features
        from features.time_context_features import add_time_context_features
        result_df = add_time_context_features(result_df)

        # Final cleanup
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.dropna()

        feature_cols = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"✅ Calculated {len(feature_cols)} technical indicators")
        print(f"Generated features: {feature_cols}")

        return result_df

    @staticmethod
    def calculate_realtime_indicators(df, last_n_rows=50):
        """Optimized calculation for real-time trading - only process recent data"""
        print("⚡ Calculating real-time indicators (optimized)...")
        
        # Use only recent data for faster processing
        recent_df = df.tail(last_n_rows).copy()
        
        # Calculate indicators on recent data
        result_df = TechnicalIndicators.calculate_volatility_indicators(recent_df)
        
        # Add custom features
        from features.custom_engineered import compute_custom_volatility_features
        result_df = compute_custom_volatility_features(result_df)
        
        # Add lagged features
        from features.lagged_features import add_volatility_lagged_features  
        result_df = add_volatility_lagged_features(result_df)
        
        # Add time context
        from features.time_context_features import add_time_context_features
        result_df = add_time_context_features(result_df)
        
        # Quick cleanup
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        result_df = result_df.dropna()
        
        print(f"⚡ Real-time calculation completed in optimized mode")
        return result_df