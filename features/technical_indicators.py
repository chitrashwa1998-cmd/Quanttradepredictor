import pandas as pd
import numpy as np
from typing import Dict, List
import ta

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

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df[high_col], df[low_col], df[close_col]).average_true_range()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(df[close_col])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        # Keltner Channel Width
        keltner = ta.volatility.KeltnerChannel(df[high_col], df[low_col], df[close_col])
        df['keltner_width'] = (keltner.keltner_channel_hband() - keltner.keltner_channel_lband()) / keltner.keltner_channel_mband()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df[close_col]).rsi()

        # Donchian Channel Width
        donchian = ta.volatility.DonchianChannel(df[high_col], df[low_col], df[close_col])
        df['donchian_width'] = (donchian.donchian_channel_hband() - donchian.donchian_channel_lband()) / df[close_col]

        return df

    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators for the dataset"""
        print("🔧 Calculating comprehensive technical indicators...")

        # Validate input data
        if not TechnicalIndicators.validate_ohlc_data(df):
            raise ValueError("Invalid OHLC data provided")

        # Create a copy to avoid modifying original data
        result_df = df.copy()

        # Calculate basic indicators
        result_df = TechnicalIndicators.calculate_volatility_indicators(result_df)

        # Add custom engineered features
        from features.custom_engineered import add_custom_features
        result_df = add_custom_features(result_df)

        # Add lagged features
        from features.lagged_features import add_lagged_features
        result_df = add_lagged_features(result_df)

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