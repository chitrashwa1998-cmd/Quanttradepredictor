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
        """Calculate all technical indicators for the dataset (excluding data leakage features)"""
        result_df = df.copy()
        
        # Price-based indicators
        result_df['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
        result_df['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
        
        result_df['ema_5'] = TechnicalIndicators.ema(df['Close'], 5)
        result_df['ema_10'] = TechnicalIndicators.ema(df['Close'], 10)
        result_df['ema_20'] = TechnicalIndicators.ema(df['Close'], 20)
        
        result_df['rsi'] = TechnicalIndicators.rsi(df['Close'])
        
        # MACD
        macd_data = TechnicalIndicators.macd(df['Close'])
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df['Close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = bb_data['upper'] - bb_data['lower']
        result_df['bb_position'] = (df['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # ATR
        result_df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        
        # Williams %R
        result_df['williams_r'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
        
        # Price ratios and differences
        result_df['high_low_ratio'] = df['High'] / df['Low']
        result_df['open_close_diff'] = df['Close'] - df['Open']
        result_df['high_close_diff'] = df['High'] - df['Close']
        result_df['close_low_diff'] = df['Close'] - df['Low']
        
        # Price momentum
        result_df['price_momentum_1'] = df['Close'].pct_change(1)
        result_df['price_momentum_3'] = df['Close'].pct_change(3)
        result_df['price_momentum_5'] = df['Close'].pct_change(5)
        
        # Volatility indicators
        result_df['volatility_10'] = df['Close'].rolling(10).std()
        result_df['volatility_20'] = df['Close'].rolling(20).std()
        
        # Additional features
        result_df['hour'] = result_df.index.hour if hasattr(result_df.index, 'hour') else 0
        
        # Technical analysis features
        result_df['Direction_Change'] = (df['Close'] > df['Open']).astype(int) != (df['Close'].shift(1) > df['Open'].shift(1)).astype(int)
        result_df['Confirmed_Bull_Reversal'] = ((df['Close'] > df['Open']) & (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['High'].shift(1))).astype(int)
        result_df['Confirmed_Bear_Reversal'] = ((df['Close'] < df['Open']) & (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'] < df['Low'].shift(1))).astype(int)
        result_df['Price'] = df['Close']
        
        # ✅ 1. CANDLE SHAPE FEATURES
        result_df['body_size'] = np.abs(df['Close'] - df['Open'])
        result_df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        result_df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        result_df['total_range'] = df['High'] - df['Low']
        result_df['body_ratio'] = result_df['body_size'] / (result_df['total_range'] + 1e-6)
        result_df['wick_ratio'] = (result_df['upper_wick'] + result_df['lower_wick']) / (result_df['body_size'] + 1e-6)
        result_df['is_bullish'] = (df['Close'] > df['Open']).astype(int)
        result_df['candle_strength'] = (df['Close'] - df['Open']) / (result_df['total_range'] + 1e-6)
        
        # ✅ 2. CANDLE PATTERN FLAGS (Binary)
        result_df['doji'] = (result_df['body_ratio'] < 0.1).astype(int)
        result_df['marubozu'] = ((result_df['upper_wick'] < 0.1 * result_df['body_size']) & 
                                (result_df['lower_wick'] < 0.1 * result_df['body_size'])).astype(int)
        result_df['hammer'] = ((result_df['lower_wick'] > 2 * result_df['body_size']) & 
                              (result_df['upper_wick'] < result_df['body_size'])).astype(int)
        result_df['shooting_star'] = ((result_df['upper_wick'] > 2 * result_df['body_size']) & 
                                     (result_df['lower_wick'] < result_df['body_size'])).astype(int)
        
        # Engulfing patterns (need previous candle data)
        prev_open = df['Open'].shift(1)
        prev_close = df['Close'].shift(1)
        result_df['engulfing_bull'] = ((df['Close'] > df['Open']) & 
                                      (df['Close'] > prev_open) & 
                                      (df['Open'] < prev_close)).astype(int)
        result_df['engulfing_bear'] = ((df['Close'] < df['Open']) & 
                                      (df['Close'] < prev_open) & 
                                      (df['Open'] > prev_close)).astype(int)
        
        # ✅ 3. CANDLE SEQUENCES
        # Bull/Bear streaks
        bullish_candles = (df['Close'] > df['Open']).astype(int)
        bearish_candles = (df['Close'] < df['Open']).astype(int)
        result_df['bull_streak_3'] = ((bullish_candles.rolling(3).sum() == 3)).astype(int)
        result_df['bear_streak_2'] = ((bearish_candles.rolling(2).sum() == 2)).astype(int)
        
        # Inside/Outside bars
        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        result_df['inside_bar'] = ((df['High'] < prev_high) & (df['Low'] > prev_low)).astype(int)
        result_df['outside_bar'] = ((df['High'] > prev_high) & (df['Low'] < prev_low)).astype(int)
        
        # Reversal bar (current candle opposite to last 2)
        current_direction = (df['Close'] > df['Open']).astype(int)
        prev_direction_1 = current_direction.shift(1)
        prev_direction_2 = current_direction.shift(2)
        result_df['reversal_bar'] = ((current_direction != prev_direction_1) & 
                                    (prev_direction_1 == prev_direction_2)).astype(int)
        
        # ✅ 4. PRICE BEHAVIOR OVER TIME
        # Dynamic threshold based on ATR for gaps
        gap_threshold = result_df['atr'] * 0.5  # Use 50% of ATR as threshold
        result_df['gap_up'] = (df['Open'] > (prev_close + gap_threshold)).astype(int)
        result_df['gap_down'] = (df['Open'] < (prev_close - gap_threshold)).astype(int)
        
        # Direction change
        current_body = df['Close'] - df['Open']
        prev_body = current_body.shift(1)
        result_df['direction_change'] = (current_body * prev_body < 0).astype(int)
        
        # Momentum surge
        rolling_avg_body_size = result_df['body_size'].rolling(20).mean()
        result_df['momentum_surge'] = (result_df['body_size'] > 1.5 * rolling_avg_body_size).astype(int)
        
        # ✅ 5. TIME-AWARE BEHAVIOR
        if hasattr(result_df.index, 'minute'):
            result_df['minute_of_hour'] = result_df.index.minute
            
            # Market timing features (assuming Indian market timings 09:15-15:30)
            market_time = result_df.index.time
            result_df['is_opening_range'] = ((result_df.index.hour == 9) & 
                                           (result_df.index.minute >= 15) & 
                                           (result_df.index.minute <= 30)).astype(int)
            result_df['is_closing_phase'] = ((result_df.index.hour == 15) & 
                                           (result_df.index.minute >= 0) & 
                                           (result_df.index.minute <= 30)).astype(int)
        else:
            # Default values if time information is not available
            result_df['minute_of_hour'] = 0
            result_df['is_opening_range'] = 0
            result_df['is_closing_phase'] = 0
        
        return result_df
