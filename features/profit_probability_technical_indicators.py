"""
Profit probability technical indicators
"""

import pandas as pd
import numpy as np
import ta
from .technical_indicators import TechnicalIndicators

class ProfitProbabilityTechnicalIndicators(TechnicalIndicators):
    def __init__(self):
        super().__init__()
    
    def calculate_profit_probability_features(self, data):
        """Calculate features specifically for profit probability prediction model"""
        if data is None or len(data) < 50:
            return None
        
        try:
            # Start with a copy of the data
            features = data.copy()
            
            # Ensure consistent column names (capital case for OHLC)
            if 'close' in features.columns and 'Close' not in features.columns:
                features = features.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            
            # Get the correct close column name
            close_col = 'Close' if 'Close' in features.columns else 'close'
            high_col = 'High' if 'High' in features.columns else 'high'
            low_col = 'Low' if 'Low' in features.columns else 'low'
            
            # Simplified profit probability-specific features
            # Price momentum for different timeframes
            features['return_1d'] = features[close_col].pct_change(1)
            features['return_3d'] = features[close_col].pct_change(3)
            features['return_5d'] = features[close_col].pct_change(5)
            
            # Volatility measures
            features['vol_5d'] = features['return_1d'].rolling(5).std()
            features['vol_10d'] = features['return_1d'].rolling(10).std()
            
            # Simple moving averages
            features['sma_10'] = features[close_col].rolling(10).mean()
            features['sma_20'] = features[close_col].rolling(20).mean()
            
            # Price relative to moving averages
            features['price_above_sma10'] = (features[close_col] > features['sma_10']).astype(int)
            features['price_above_sma20'] = (features[close_col] > features['sma_20']).astype(int)
            
            # Simple RSI
            delta = features[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            features['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Target: Future profit probability (2% profit in 3 days)
            future_return = features[close_col].shift(-3) / features[close_col] - 1
            features['profit_target'] = (future_return > 0.02).astype(int)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            print(f"Error calculating profit probability features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def add_profit_features(self, df):
        """Add features related to profit potential"""
        # Get the correct close column name
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Price momentum for different timeframes
        df['return_1d'] = df[close_col].pct_change(1)
        df['return_3d'] = df[close_col].pct_change(3)
        df['return_5d'] = df[close_col].pct_change(5)
        df['return_10d'] = df[close_col].pct_change(10)
        
        # Volatility-adjusted returns
        df['vol_5d'] = df['return_1d'].rolling(5).std()
        df['vol_10d'] = df['return_1d'].rolling(10).std()
        df['sharpe_5d'] = df['return_1d'].rolling(5).mean() / df['vol_5d']
        df['sharpe_10d'] = df['return_1d'].rolling(10).mean() / df['vol_10d']
        
        # Trend strength
        df['trend_strength'] = abs(df[close_col].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0]))
        
        # Price efficiency
        df['price_efficiency'] = abs(df[close_col] - df[close_col].shift(10)) / (df[high_col].rolling(10).max() - df[low_col].rolling(10).min())
        
        return df
    
    def add_risk_features(self, df):
        """Add risk-related features"""
        # Get the correct close column name
        close_col = 'Close' if 'Close' in df.columns else 'close'
        
        # Drawdown measures
        df['rolling_max'] = df[close_col].rolling(window=20).max()
        df['drawdown'] = (df[close_col] - df['rolling_max']) / df['rolling_max']
        df['max_drawdown_5d'] = df['drawdown'].rolling(5).min()
        df['max_drawdown_10d'] = df['drawdown'].rolling(10).min()
        
        # Value at Risk (simplified)
        df['var_95'] = df['return_1d'].rolling(20).quantile(0.05)
        df['var_99'] = df['return_1d'].rolling(20).quantile(0.01)
        
        # Tail risk
        df['skewness'] = df['return_1d'].rolling(20).skew()
        df['kurtosis'] = df['return_1d'].rolling(20).kurt()
        
        # Volatility clustering
        df['vol_clustering'] = df['vol_5d'] / df['vol_10d']
        
        return df
    
    def add_market_regime_features(self, df):
        """Add market regime identification features"""
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        # Regime indicators
        df['high_vol_regime'] = (df['vol_10d'] > df['vol_10d'].rolling(50).quantile(0.75)).astype(int)
        df['trending_regime'] = (df['trend_strength'] > df['trend_strength'].rolling(50).quantile(0.75)).astype(int)
        
        # Market stress indicators
        df['stress_indicator'] = ((df['vol_5d'] > df['vol_5d'].rolling(20).quantile(0.9)) & 
                                 (df['drawdown'] < -0.05)).astype(int)
        
        # Momentum regime
        df['momentum_regime'] = ((df['return_5d'] > 0) & (df['return_10d'] > 0)).astype(int)
        
        # Range-bound detection
        df['range_bound'] = ((df[high_col].rolling(20).max() - df[low_col].rolling(20).min()) / 
                            df[close_col].rolling(20).mean() < 0.1).astype(int)
        
        return df
    
    def add_timing_features(self, df):
        """Add timing-related features"""
        # Entry timing - create default series if columns don't exist
        rsi_14_series = df.get('rsi_14', pd.Series([50] * len(df), index=df.index))
        bb_position_series = df.get('bb_position', pd.Series([0.5] * len(df), index=df.index))
        
        df['rsi_entry'] = ((rsi_14_series < 35) | (rsi_14_series > 65)).astype(int)
        df['bb_entry'] = ((bb_position_series < 0.2) | (bb_position_series > 0.8)).astype(int)
        
        # Confluence signals - create default series if columns don't exist
        price_above_sma20_series = df.get('price_above_sma20', pd.Series([0] * len(df), index=df.index))
        macd_bullish_series = df.get('macd_bullish', pd.Series([0] * len(df), index=df.index))
        rsi_14_series = df.get('rsi_14', pd.Series([50] * len(df), index=df.index))
        
        df['bullish_confluence'] = ((price_above_sma20_series == 1) & 
                                   (macd_bullish_series == 1) & 
                                   (rsi_14_series > 50)).astype(int)
        df['bearish_confluence'] = ((price_above_sma20_series == 0) & 
                                   (macd_bullish_series == 0) & 
                                   (rsi_14_series < 50)).astype(int)
        
        # Breakout conditions
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        
        if 'recent_high' in df.columns:
            df['breakout_high'] = (df[close_col] > df['recent_high'].shift(1)).astype(int)
        else:
            # Create recent_high feature
            df['recent_high'] = df[high_col].rolling(20).max()
            df['breakout_high'] = (df[close_col] > df['recent_high'].shift(1)).astype(int)
            
        if 'recent_low' in df.columns:
            df['breakout_low'] = (df[close_col] < df['recent_low'].shift(1)).astype(int)
        else:
            # Create recent_low feature
            df['recent_low'] = df[low_col].rolling(20).min()
            df['breakout_low'] = (df[close_col] < df['recent_low'].shift(1)).astype(int)
        
        return df
    
    def calculate_profit_targets(self, df, holding_periods=[1, 3, 5]):
        """Calculate profit probability targets for different holding periods"""
        close_col = 'Close' if 'Close' in df.columns else 'close'
        
        for period in holding_periods:
            # Future returns
            future_return = df[close_col].shift(-period) / df[close_col] - 1
            
            # Binary profit targets (different thresholds)
            df[f'profit_1pct_{period}d'] = (future_return > 0.01).astype(int)
            df[f'profit_2pct_{period}d'] = (future_return > 0.02).astype(int)
            df[f'profit_3pct_{period}d'] = (future_return > 0.03).astype(int)
            
            # Risk-adjusted profit (profit above risk threshold)
            risk_threshold = df['vol_5d'] * np.sqrt(period)
            df[f'risk_adj_profit_{period}d'] = (future_return > risk_threshold).astype(int)
        
        # Main target: 2% profit in 3 days
        df['profit_target'] = df['profit_2pct_3d']
        
        return df