
import pandas as pd
import numpy as np
from typing import Dict, List
import ta
import pandas_ta as pta
from finta import TA

class TechnicalIndicators:
    """Calculate model-specific technical indicators for trading analysis."""

    @staticmethod
    def calculate_volatility_indicators(df):
        """Calculate indicators specifically for volatility model"""
        df = df.copy()

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        # Keltner Channel Width
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['keltner_width'] = (keltner.keltner_channel_hband() - keltner.keltner_channel_lband()) / keltner.keltner_channel_mband()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Donchian Channel Width
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['donchian_width'] = (donchian.donchian_channel_hband() - donchian.donchian_channel_lband()) / df['Close']

        return df

    @staticmethod
    def calculate_direction_indicators(df):
        """Calculate indicators specifically for direction model"""
        df = df.copy()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # EMA fast and slow
        df['ema_fast'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()

        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

        # OBV
        if 'Volume' in df.columns:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        return df

    @staticmethod
    def calculate_magnitude_indicators(df):
        """Calculate indicators specifically for magnitude model"""
        df = df.copy()

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        # EMA Deviation
        df['ema_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
        df['ema_deviation'] = abs(df['Close'] - df['ema_20']) / df['ema_20']

        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['donchian_high'] = donchian.donchian_channel_hband()
        df['donchian_low'] = donchian.donchian_channel_lband()
        df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / df['Close']

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # MACD Histogram
        macd = ta.trend.MACD(df['Close'])
        df['macd_histogram'] = macd.macd_diff()

        return df

    @staticmethod
    def calculate_profit_probability_indicators(df):
        """Calculate indicators specifically for profit probability model"""
        df = df.copy()

        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

        # EMA crossover
        df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        df['ema_crossover'] = (df['ema_12'] > df['ema_26']).astype(int)

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        return df

    @staticmethod
    def calculate_trend_indicators(df):
        """Calculate indicators specifically for trend classification model"""
        df = df.copy()

        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()

        # Donchian Channels
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['donchian_high'] = donchian.donchian_channel_hband()
        df['donchian_low'] = donchian.donchian_channel_lband()

        # EMA fast/slow
        df['ema_fast'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()

        # MACD Histogram
        macd = ta.trend.MACD(df['Close'])
        df['macd_histogram'] = macd.macd_diff()

        # OBV
        if 'Volume' in df.columns:
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

        return df

    @staticmethod
    def calculate_reversal_indicators(df):
        """Calculate indicators specifically for reversal points model"""
        df = df.copy()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Bollinger Bands (upper/lower hits)
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_upper_hit'] = (df['Close'] >= df['bb_upper']).astype(int)
        df['bb_lower_hit'] = (df['Close'] <= df['bb_lower']).astype(int)

        # MACD Histogram
        macd = ta.trend.MACD(df['Close'])
        df['macd_histogram'] = macd.macd_diff()

        return df

    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all model-specific indicators for the dataset
        This replaces the old 58-indicator system with targeted indicators
        """
        df = df.copy()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        try:
            # Calculate all model-specific indicators
            df = TechnicalIndicators.calculate_volatility_indicators(df)
            df = TechnicalIndicators.calculate_direction_indicators(df)
            df = TechnicalIndicators.calculate_magnitude_indicators(df)
            df = TechnicalIndicators.calculate_profit_probability_indicators(df)
            df = TechnicalIndicators.calculate_trend_indicators(df)
            df = TechnicalIndicators.calculate_reversal_indicators(df)

            # Remove duplicate columns (some indicators are used in multiple models)
            df = df.loc[:, ~df.columns.duplicated()]

            return df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df
