import pandas as pd
import numpy as np
from typing import Dict, List
import ta

class TechnicalIndicators:
    """Calculate only the 5 specified technical indicators for volatility-focused trading analysis."""

    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate ONLY the 5 specified indicators:
        1. ATR (Average True Range)
        2. Bollinger Band Width  
        3. Keltner Channel Width
        4. RSI (for regime classification)
        5. Donchian Channel Width
        """
        df = df.copy()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        try:
            # 1. ATR (Average True Range)
            df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

            # 2. Bollinger Band Width
            bb = ta.volatility.BollingerBands(df['Close'])
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_middle = bb.bollinger_mavg()
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # 3. Keltner Channel Width
            keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            keltner_upper = keltner.keltner_channel_hband()
            keltner_lower = keltner.keltner_channel_lband()
            keltner_middle = keltner.keltner_channel_mband()
            df['keltner_width'] = (keltner_upper - keltner_lower) / keltner_middle

            # 4. RSI (for regime classification)
            df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

            # 5. Donchian Channel Width
            donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
            donchian_upper = donchian.donchian_channel_hband()
            donchian_lower = donchian.donchian_channel_lband()
            df['donchian_width'] = (donchian_upper - donchian_lower) / df['Close']

            print(f"✅ Calculated 5 volatility-focused indicators: ATR, BB_width, Keltner_width, RSI, Donchian_width")

            return df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df

    @staticmethod
    def calculate_volatility_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)

    @staticmethod
    def calculate_direction_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)

    @staticmethod
    def calculate_magnitude_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)

    @staticmethod
    def calculate_profit_probability_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)

    @staticmethod
    def calculate_trend_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)

    @staticmethod
    def calculate_reversal_indicators(df):
        """Legacy method - now uses the same 5 indicators"""
        return TechnicalIndicators.calculate_all_indicators(df)