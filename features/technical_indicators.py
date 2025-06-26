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
        """
        Calculate volatility-specific indicators for the dataset
        Only includes: ATR, Bollinger Band Width, Keltner Channel Width, RSI, Donchian Channel Width
        """
        df = df.copy()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        try:
            # Calculate volatility-specific indicators only
            df = TechnicalIndicators.calculate_volatility_indicators(df)

            # Remove any duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]

            print(f"✅ Calculated {len([col for col in df.columns if col not in required_columns])} volatility-focused indicators: ATR, BB_width, Keltner_width, RSI, Donchian_width")

            return df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df