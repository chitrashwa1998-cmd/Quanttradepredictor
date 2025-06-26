import pandas as pd
import numpy as np
from typing import Dict, List
import ta
import pandas_ta as pta
from finta import TA

class TechnicalIndicators:
    """Calculate various technical indicators for trading analysis."""

    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()

    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()

        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()

        return {
            'k': k_percent,
            'd': d_percent
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv_values = []
        obv_val = 0

        for i in range(len(close)):
            if i == 0:
                obv_val = volume.iloc[i]
            else:
                if close.iloc[i] > close.iloc[i-1]:
                    obv_val += volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv_val -= volume.iloc[i]
            obv_values.append(obv_val)

        return pd.Series(obv_values, index=close.index)

    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all technical indicators for the dataset
        """
        df = df.copy()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        try:
            # Price-based indicators
            df = TechnicalIndicators._calculate_price_indicators(df)

            # Volume indicators (if volume is available)
            if 'Volume' in df.columns:
                df = TechnicalIndicators._calculate_volume_indicators(df)

            # Momentum indicators
            df = TechnicalIndicators._calculate_momentum_indicators(df)

            # Volatility indicators
            df = TechnicalIndicators._calculate_volatility_indicators(df)

            # Trend indicators
            df = TechnicalIndicators._calculate_trend_indicators(df)

            # Support and resistance
            df = TechnicalIndicators._calculate_support_resistance(df)

            # Pattern recognition
            df = TechnicalIndicators._calculate_pattern_features(df)

            # Time-based features
            df = TechnicalIndicators._calculate_time_features(df)

            # Custom engineered features
            df = TechnicalIndicators._calculate_custom_features(df)

            return df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return df

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