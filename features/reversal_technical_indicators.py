"""
Reversal prediction technical indicators
"""

import pandas as pd
import numpy as np
import ta
from .technical_indicators import TechnicalIndicators

class ReversalTechnicalIndicators(TechnicalIndicators):
    def __init__(self):
        super().__init__()
    
    def calculate_reversal_features(self, data):
        """Calculate features specifically for reversal prediction model"""
        if data is None or len(data) < 50:
            return None
        
        try:
            # Start with base features
            features = self.calculate_base_features(data)
            
            # Reversal-specific features
            features = self.add_reversal_patterns(features)
            features = self.add_divergence_features(features)
            features = self.add_extreme_conditions(features)
            features = self.add_support_resistance(features)
            
            # Target: Reversal in next few periods
            features = self.calculate_reversal_targets(features)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            print(f"Error calculating reversal features: {e}")
            return None
    
    def add_reversal_patterns(self, df):
        """Add candlestick reversal pattern features"""
        # Basic reversal patterns
        df['bullish_hammer'] = ((df['close'] > df['open']) & 
                               ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) &
                               ((df['high'] - df['close']) < 0.3 * (df['close'] - df['open']))).astype(int)
        
        df['bearish_shooting_star'] = ((df['open'] > df['close']) & 
                                      ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) &
                                      ((df['close'] - df['low']) < 0.3 * (df['open'] - df['close']))).astype(int)
        
        df['doji_reversal'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                  (df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['close'] > df['open'].shift(1)) & 
                                  (df['open'] < df['close'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                  (df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['close'] < df['open'].shift(1)) & 
                                  (df['open'] > df['close'].shift(1))).astype(int)
        
        # Piercing line / Dark cloud cover
        df['piercing_line'] = ((df['close'] > df['open']) & 
                              (df['close'].shift(1) < df['open'].shift(1)) &
                              (df['open'] < df['close'].shift(1)) & 
                              (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)).astype(int)
        
        df['dark_cloud'] = ((df['close'] < df['open']) & 
                           (df['close'].shift(1) > df['open'].shift(1)) &
                           (df['open'] > df['close'].shift(1)) & 
                           (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2)).astype(int)
        
        return df
    
    def add_divergence_features(self, df):
        """Add momentum divergence features"""
        # RSI divergence
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['price_higher_high'] = ((df['high'] > df['high'].shift(5)) & 
                                  (df['high'].shift(5) > df['high'].shift(10))).astype(int)
        df['price_lower_low'] = ((df['low'] < df['low'].shift(5)) & 
                                (df['low'].shift(5) < df['low'].shift(10))).astype(int)
        df['rsi_lower_high'] = ((df['rsi_14'] < df['rsi_14'].shift(5)) & 
                               (df['rsi_14'].shift(5) < df['rsi_14'].shift(10))).astype(int)
        df['rsi_higher_low'] = ((df['rsi_14'] > df['rsi_14'].shift(5)) & 
                               (df['rsi_14'].shift(5) > df['rsi_14'].shift(10))).astype(int)
        
        # Bearish divergence (price makes higher highs, RSI makes lower highs)
        df['bearish_divergence'] = (df['price_higher_high'] & df['rsi_lower_high']).astype(int)
        
        # Bullish divergence (price makes lower lows, RSI makes higher lows)
        df['bullish_divergence'] = (df['price_lower_low'] & df['rsi_higher_low']).astype(int)
        
        # MACD divergence
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_lower_high'] = ((df['macd'] < df['macd'].shift(5)) & 
                                (df['macd'].shift(5) < df['macd'].shift(10))).astype(int)
        df['macd_higher_low'] = ((df['macd'] > df['macd'].shift(5)) & 
                                (df['macd'].shift(5) > df['macd'].shift(10))).astype(int)
        
        df['macd_bearish_div'] = (df['price_higher_high'] & df['macd_lower_high']).astype(int)
        df['macd_bullish_div'] = (df['price_lower_low'] & df['macd_higher_low']).astype(int)
        
        return df
    
    def add_extreme_conditions(self, df):
        """Add overbought/oversold extreme condition features"""
        # RSI extremes
        df['rsi_oversold'] = (df['rsi_14'] < 20).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 80).astype(int)
        df['rsi_extreme_oversold'] = (df['rsi_14'] < 15).astype(int)
        df['rsi_extreme_overbought'] = (df['rsi_14'] > 85).astype(int)
        
        # Stochastic extremes
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_oversold'] = (df['stoch_k'] < 15).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 85).astype(int)
        
        # Williams %R extremes
        willr = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'])
        df['willr'] = willr.williams_r()
        df['willr_oversold'] = (df['willr'] < -85).astype(int)
        df['willr_overbought'] = (df['willr'] > -15).astype(int)
        
        # Multiple oscillator confirmation
        df['multiple_oversold'] = (df['rsi_oversold'] + df['stoch_oversold'] + df['willr_oversold'] >= 2).astype(int)
        df['multiple_overbought'] = (df['rsi_overbought'] + df['stoch_overbought'] + df['willr_overbought'] >= 2).astype(int)
        
        return df
    
    def add_support_resistance(self, df):
        """Add support/resistance level features"""
        # Pivot points (simplified)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['support1'] = 2 * df['pivot'] - df['high']
        df['resistance1'] = 2 * df['pivot'] - df['low']
        
        # Distance to support/resistance
        df['dist_to_support'] = (df['close'] - df['support1']) / df['close']
        df['dist_to_resistance'] = (df['resistance1'] - df['close']) / df['close']
        
        # Near support/resistance
        df['near_support'] = (abs(df['dist_to_support']) < 0.01).astype(int)
        df['near_resistance'] = (abs(df['dist_to_resistance']) < 0.01).astype(int)
        
        # Dynamic support/resistance (recent highs/lows)
        df['recent_high_20'] = df['high'].rolling(20).max()
        df['recent_low_20'] = df['low'].rolling(20).min()
        df['at_resistance'] = (df['close'] >= df['recent_high_20'] * 0.998).astype(int)
        df['at_support'] = (df['close'] <= df['recent_low_20'] * 1.002).astype(int)
        
        # Fibonacci retracements (simplified)
        df['fib_range'] = df['recent_high_20'] - df['recent_low_20']
        df['fib_23'] = df['recent_high_20'] - 0.236 * df['fib_range']
        df['fib_38'] = df['recent_high_20'] - 0.382 * df['fib_range']
        df['fib_50'] = df['recent_high_20'] - 0.500 * df['fib_range']
        df['fib_62'] = df['recent_high_20'] - 0.618 * df['fib_range']
        
        df['near_fib_level'] = ((abs(df['close'] - df['fib_23']) < df['close'] * 0.005) |
                               (abs(df['close'] - df['fib_38']) < df['close'] * 0.005) |
                               (abs(df['close'] - df['fib_50']) < df['close'] * 0.005) |
                               (abs(df['close'] - df['fib_62']) < df['close'] * 0.005)).astype(int)
        
        return df
    
    def calculate_reversal_targets(self, df):
        """Calculate reversal targets for different timeframes"""
        # Short-term reversal (1-3 periods)
        df['reversal_1d'] = self._calculate_reversal_target(df, 1)
        df['reversal_3d'] = self._calculate_reversal_target(df, 3)
        df['reversal_5d'] = self._calculate_reversal_target(df, 5)
        
        # Main target: 3-day reversal
        df['reversal_target'] = df['reversal_3d']
        
        return df
    
    def _calculate_reversal_target(self, df, periods):
        """Calculate if a reversal occurs within the specified periods"""
        reversal_signals = []
        
        for i in range(len(df)):
            if i + periods >= len(df):
                reversal_signals.append(0)
                continue
                
            current_close = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+periods+1]
            
            # Check for significant reversal (>2% move in opposite direction of recent trend)
            recent_trend = df['close'].iloc[i] - df['close'].iloc[max(0, i-5)]
            
            if recent_trend > 0:  # Recent uptrend
                # Look for downward reversal
                min_future = future_prices.min()
                reversal = (current_close - min_future) / current_close > 0.02
            else:  # Recent downtrend
                # Look for upward reversal
                max_future = future_prices.max()
                reversal = (max_future - current_close) / current_close > 0.02
            
            reversal_signals.append(int(reversal))
        
        return reversal_signals