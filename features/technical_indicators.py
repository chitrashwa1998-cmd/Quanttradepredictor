import pandas as pd
import numpy as np
from typing import Dict, List

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
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low

        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus < 0] = 0
        dm_plus[(dm_plus - dm_minus) <= 0] = 0
        dm_minus[(dm_minus - dm_plus) <= 0] = 0

        # Smoothed values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()

        # Directional Indicators
        di_plus = (dm_plus_smooth / tr_smooth) * 100
        di_minus = (dm_minus_smooth / tr_smooth) * 100

        # ADX
        dx = (np.abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
        adx = dx.rolling(window=window).mean()

        return adx.fillna(25.0)  # Fill NaN with neutral value

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.fillna(0.0)  # Fill NaN with neutral value

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
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all 36 specified technical indicators"""
        result_df = df.copy()

        # Price-based Moving Averages (7 indicators)
        result_df['sma_5'] = TechnicalIndicators.sma(df['Close'], 5)
        result_df['sma_10'] = TechnicalIndicators.sma(df['Close'], 10)
        result_df['sma_20'] = TechnicalIndicators.sma(df['Close'], 20)
        result_df['sma_50'] = TechnicalIndicators.sma(df['Close'], 50)

        result_df['ema_5'] = TechnicalIndicators.ema(df['Close'], 5)
        result_df['ema_10'] = TechnicalIndicators.ema(df['Close'], 10)
        result_df['ema_20'] = TechnicalIndicators.ema(df['Close'], 20)

        # Momentum Indicators (9 indicators)
        result_df['rsi'] = TechnicalIndicators.rsi(df['Close'])

        # MACD components
        macd_data = TechnicalIndicators.macd(df['Close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']

        # Price momentum
        result_df['price_momentum_1'] = df['Close'].pct_change(1)
        result_df['price_momentum_3'] = df['Close'].pct_change(3)
        result_df['price_momentum_5'] = df['Close'].pct_change(5)
        result_df['price_momentum_10'] = df['Close'].pct_change(10)

        # Williams %R
        result_df['williams_r'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])

        # Volatility Indicators (4 indicators)
        result_df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        result_df['volatility_10'] = df['Close'].rolling(10).std()
        result_df['volatility_20'] = df['Close'].rolling(20).std()

        # Bollinger Bands (5 indicators)
        bb_data = TechnicalIndicators.bollinger_bands(df['Close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = bb_data['upper'] - bb_data['lower']
        result_df['bb_position'] = (df['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])

        # Stochastic Oscillator (2 indicators)
        stoch_data = TechnicalIndicators.stochastic(df['High'], df['Low'], df['Close'])
        result_df['stoch_k'] = stoch_data['k']
        result_df['stoch_d'] = stoch_data['d']

        # Price Relationships (4 indicators)
        result_df['high_low_ratio'] = df['High'] / df['Low']
        result_df['open_close_diff'] = df['Close'] - df['Open']
        result_df['high_close_diff'] = df['High'] - df['Close']
        result_df['close_low_diff'] = df['Close'] - df['Low']

        # Volume Indicators (3 indicators)
        if 'Volume' in df.columns:
            result_df['volume_sma'] = TechnicalIndicators.sma(df['Volume'], 10)
            result_df['volume_ratio'] = df['Volume'] / result_df['volume_sma']
            result_df['obv'] = TechnicalIndicators.obv(df['Close'], df['Volume'])
        else:
            # Create dummy volume indicators if Volume column is missing
            result_df['volume_sma'] = 1000000.0
            result_df['volume_ratio'] = 1.0
            result_df['obv'] = 0.0

        # Time-based Features (3 indicators)
        result_df['day_of_week'] = result_df.index.dayofweek if hasattr(result_df.index, 'dayofweek') else 0
        result_df['month'] = result_df.index.month if hasattr(result_df.index, 'month') else 1
        result_df['hour'] = result_df.index.hour if hasattr(result_df.index, 'hour') else 0

        return result_df