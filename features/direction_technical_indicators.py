"""
Direction prediction technical indicators
"""

import pandas as pd
import numpy as np
import ta
from .technical_indicators import TechnicalIndicators

class DirectionTechnicalIndicators(TechnicalIndicators):
    def __init__(self):
        super().__init__()
    
    def calculate_direction_features(self, data):
        """Calculate features specifically for direction prediction model"""
        if data is None or len(data) < 50:
            return None
        
        try:
            # Start with base features
            features = self.calculate_base_features(data)
            
            # Direction-specific features
            features = self.add_momentum_features(features)
            features = self.add_trend_features(features)
            features = self.add_volume_features(features)
            features = self.add_price_action_features(features)
            
            # Target: Next candle direction (1 for up, 0 for down)
            features['direction_target'] = (features['close'].shift(-1) > features['close']).astype(int)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            print(f"Error calculating direction features: {e}")
            return None
    
    def add_momentum_features(self, df):
        """Add momentum-based features for direction prediction"""
        # RSI variations
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        return df
    
    def add_trend_features(self, df):
        """Add trend-based features"""
        # Moving averages
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # Price vs MA positions
        df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        df['sma20_above_sma50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']).rolling(10).std()
        
        return df
    
    def add_volume_features(self, df):
        """Add volume-based features"""
        if 'volume' in df.columns:
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_increasing'] = (df['volume'] > df['volume'].shift(1)).astype(int)
            
            # On-Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_slope'] = df['obv'].diff(5)
            
            # Accumulation/Distribution Line
            df['ad_line'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
        else:
            # If no volume data, create neutral features
            df['volume_ratio'] = 1.0
            df['volume_increasing'] = 0
            df['obv_slope'] = 0
            df['ad_line'] = 0
        
        return df
    
    def add_price_action_features(self, df):
        """Add price action features"""
        # Candle patterns
        df['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) &
                       ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open']))).astype(int)
        df['shooting_star'] = ((df['open'] > df['close']) & 
                              ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) &
                              ((df['close'] - df['low']) < 0.1 * (df['open'] - df['close']))).astype(int)
        
        # Price momentum
        df['momentum_1'] = df['close'] / df['close'].shift(1) - 1
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        # Support/Resistance levels
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()
        df['distance_to_high'] = (df['recent_high'] - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['recent_low']) / df['close']
        
        return df