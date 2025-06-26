import pandas as pd
import numpy as np
from typing import Dict, List
import ta

class TechnicalIndicators:
    """Calculate volatility-specific technical indicators for trading analysis."""

    @staticmethod
    def calculate_volatility_indicators(df):
        """Calculate only the required volatility indicators"""
        df = df.copy()

        # ATR - Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['Close'])
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_middle = bb.bollinger_mavg()
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower

        # Keltner Channel Width
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        keltner_upper = keltner.keltner_channel_hband()
        keltner_lower = keltner.keltner_channel_lband()
        keltner_middle = keltner.keltner_channel_mband()
        df['keltner_width'] = (keltner_upper - keltner_lower) / keltner_middle

        # RSI for regime classification
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Donchian Channel Width
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        donchian_upper = donchian.donchian_channel_hband()
        donchian_lower = donchian.donchian_channel_lband()
        df['donchian_width'] = (donchian_upper - donchian_lower) / df['Close']

        return df

    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate only the volatility-specific indicators
        """
        df = df.copy()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        try:
            # Calculate only volatility indicators
            df = TechnicalIndicators.calculate_volatility_indicators(df)
            return df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df