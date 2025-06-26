
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class TimeContextFeatures:
    """Generate time-based and contextual features for trading analysis."""
    
    @staticmethod
    def market_session_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market session and timing features"""
        result_df = df.copy()
        
        # Ensure datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)
        
        # Basic time features
        result_df['hour'] = result_df.index.hour
        result_df['minute'] = result_df.index.minute
        result_df['day_of_week'] = result_df.index.dayofweek
        result_df['day_of_month'] = result_df.index.day
        result_df['month'] = result_df.index.month
        result_df['quarter'] = result_df.index.quarter
        result_df['is_weekend'] = (result_df.index.dayofweek >= 5).astype(int)
        
        # Market session timing (Indian market: 9:15 AM - 3:30 PM)
        result_df['is_pre_market'] = ((result_df['hour'] < 9) | 
                                     ((result_df['hour'] == 9) & (result_df['minute'] < 15))).astype(int)
        result_df['is_opening_hour'] = ((result_df['hour'] == 9) | 
                                       ((result_df['hour'] == 10) & (result_df['minute'] < 30))).astype(int)
        result_df['is_mid_session'] = ((result_df['hour'] >= 11) & (result_df['hour'] <= 13)).astype(int)
        result_df['is_closing_hour'] = ((result_df['hour'] == 15) | 
                                       ((result_df['hour'] == 14) & (result_df['minute'] >= 30))).astype(int)
        result_df['is_post_market'] = (result_df['hour'] > 15).astype(int)
        
        # Minutes since market open
        market_open_minutes = (result_df['hour'] - 9) * 60 + (result_df['minute'] - 15)
        result_df['minutes_from_open'] = np.where(market_open_minutes >= 0, market_open_minutes, 0)
        
        # Minutes until market close
        market_close_minutes = (15 - result_df['hour']) * 60 + (30 - result_df['minute'])
        result_df['minutes_to_close'] = np.where(market_close_minutes >= 0, market_close_minutes, 0)
        
        # Session progress (0-1)
        total_market_minutes = 6 * 60 + 15  # 6 hours 15 minutes
        result_df['session_progress'] = result_df['minutes_from_open'] / total_market_minutes
        result_df['session_progress'] = np.clip(result_df['session_progress'], 0, 1)
        
        return result_df
    
    @staticmethod
    def cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding of time features"""
        result_df = df.copy()
        
        # Ensure datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)
        
        # Cyclical encoding for time features
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df.index.hour / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df.index.hour / 24)
        result_df['minute_sin'] = np.sin(2 * np.pi * result_df.index.minute / 60)
        result_df['minute_cos'] = np.cos(2 * np.pi * result_df.index.minute / 60)
        result_df['day_sin'] = np.sin(2 * np.pi * result_df.index.dayofweek / 7)
        result_df['day_cos'] = np.cos(2 * np.pi * result_df.index.dayofweek / 7)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df.index.month / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df.index.month / 12)
        
        return result_df
    
    @staticmethod
    def volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime and market condition features"""
        result_df = df.copy()
        
        # Volatility percentiles
        result_df['volatility_5'] = result_df['Close'].rolling(5).std()
        result_df['volatility_20'] = result_df['Close'].rolling(20).std()
        result_df['volatility_50'] = result_df['Close'].rolling(50).std()
        
        # Volatility regime classification
        vol_20_quantiles = result_df['volatility_20'].rolling(252).quantile([0.25, 0.5, 0.75])
        result_df['vol_regime_low'] = (result_df['volatility_20'] <= vol_20_quantiles.iloc[:, 0]).astype(int)
        result_df['vol_regime_medium'] = ((result_df['volatility_20'] > vol_20_quantiles.iloc[:, 0]) & 
                                         (result_df['volatility_20'] <= vol_20_quantiles.iloc[:, 2])).astype(int)
        result_df['vol_regime_high'] = (result_df['volatility_20'] > vol_20_quantiles.iloc[:, 2]).astype(int)
        
        # Volatility trend
        result_df['vol_trend_5'] = (result_df['volatility_5'] / result_df['volatility_20'] - 1) * 100
        result_df['vol_expansion'] = (result_df['volatility_20'] > result_df['volatility_50']).astype(int)
        result_df['vol_contraction'] = (result_df['volatility_20'] < result_df['volatility_50']).astype(int)
        
        return result_df
    
    @staticmethod
    def market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure and liquidity features"""
        result_df = df.copy()
        
        # Price impact and efficiency measures
        result_df['price_impact'] = np.abs(result_df['Close'] - result_df['Open']) / result_df['Open']
        result_df['intraday_return'] = (result_df['Close'] - result_df['Open']) / result_df['Open']
        result_df['overnight_return'] = (result_df['Open'] - result_df['Close'].shift(1)) / result_df['Close'].shift(1)
        
        # Market efficiency measures
        result_df['price_efficiency'] = np.abs(result_df['Close'].pct_change().rolling(5).mean())
        result_df['return_autocorr'] = result_df['Close'].pct_change().rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 1 else 0
        )
        
        # Liquidity proxies
        if 'Volume' in result_df.columns:
            result_df['dollar_volume'] = result_df['Close'] * result_df['Volume']
            result_df['volume_price_trend'] = (result_df['Close'].diff() * result_df['Volume']).rolling(5).sum()
            result_df['volume_weighted_price'] = (result_df['Close'] * result_df['Volume']).rolling(20).sum() / result_df['Volume'].rolling(20).sum()
        else:
            result_df['dollar_volume'] = 0
            result_df['volume_price_trend'] = 0
            result_df['volume_weighted_price'] = result_df['Close']
        
        return result_df
    
    @staticmethod
    def economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add economic calendar and event-based features"""
        result_df = df.copy()
        
        # Ensure datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            result_df.index = pd.to_datetime(result_df.index)
        
        # Month-end and quarter-end effects
        result_df['is_month_end'] = (result_df.index.day >= 25).astype(int)
        result_df['is_quarter_end'] = ((result_df.index.month % 3 == 0) & 
                                      (result_df.index.day >= 25)).astype(int)
        result_df['is_year_end'] = ((result_df.index.month == 12) & 
                                   (result_df.index.day >= 25)).astype(int)
        
        # Holiday effects (simplified - can be expanded with actual holiday calendar)
        result_df['is_friday'] = (result_df.index.dayofweek == 4).astype(int)
        result_df['is_monday'] = (result_df.index.dayofweek == 0).astype(int)
        
        # Option expiry effects (monthly - third Thursday)
        result_df['days_to_month_end'] = (result_df.index + pd.offsets.MonthEnd(0) - result_df.index).days
        result_df['is_option_expiry_week'] = (result_df['days_to_month_end'] <= 7).astype(int)
        
        return result_df
    
    @staticmethod
    def generate_all_time_context_features(df: pd.DataFrame) -> pd.DataFrame:
        """Generate all time and context-based features"""
        result_df = df.copy()
        
        # Apply all feature engineering methods
        result_df = TimeContextFeatures.market_session_features(result_df)
        result_df = TimeContextFeatures.cyclical_time_features(result_df)
        result_df = TimeContextFeatures.volatility_regime_features(result_df)
        result_df = TimeContextFeatures.market_microstructure_features(result_df)
        result_df = TimeContextFeatures.economic_calendar_features(result_df)
        
        return result_df
