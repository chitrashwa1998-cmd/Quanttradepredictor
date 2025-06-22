import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Optional

class TechnicalIndicators:
    """Technical indicators calculator for trading strategies."""

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators optimized for 5-minute scalping."""

        if df.empty:
            raise ValueError("DataFrame is empty")

        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise ValueError("DataFrame must contain OHLC columns")

        # Work on a copy
        result_df = df.copy()

        # Ensure we have enough data for all indicators
        if len(result_df) < 50:
            raise ValueError("Need at least 50 data points for technical indicators")

        try:
            # 1. Simple Moving Averages (3 indicators)
            result_df['sma_5'] = ta.SMA(result_df['Close'], timeperiod=5)
            result_df['sma_10'] = ta.SMA(result_df['Close'], timeperiod=10)
            result_df['sma_20'] = ta.SMA(result_df['Close'], timeperiod=20)

            # 2. Exponential Moving Averages (3 indicators)
            result_df['ema_5'] = ta.EMA(result_df['Close'], timeperiod=5)
            result_df['ema_10'] = ta.EMA(result_df['Close'], timeperiod=10)
            result_df['ema_20'] = ta.EMA(result_df['Close'], timeperiod=20)

            # 3. RSI (1 indicator)
            result_df['rsi'] = ta.RSI(result_df['Close'], timeperiod=14)

            # 4. MACD (3 indicators)
            macd, macd_signal, macd_histogram = ta.MACD(result_df['Close'])
            result_df['macd'] = macd
            result_df['macd_signal'] = macd_signal
            result_df['macd_histogram'] = macd_histogram

            # 5. Bollinger Bands (3 indicators)
            bb_upper, bb_middle, bb_lower = ta.BBANDS(result_df['Close'], timeperiod=20)
            result_df['bb_upper'] = bb_upper
            result_df['bb_lower'] = bb_lower
            result_df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # 6. ATR (1 indicator)
            result_df['atr'] = ta.ATR(result_df['High'], result_df['Low'], result_df['Close'], timeperiod=14)

            # 7. ADX (1 indicator)
            result_df['adx'] = ta.ADX(result_df['High'], result_df['Low'], result_df['Close'], timeperiod=14)

            # 8. CCI (1 indicator)
            result_df['cci'] = ta.CCI(result_df['High'], result_df['Low'], result_df['Close'], timeperiod=14)

            # 9. Williams %R (1 indicator)
            result_df['williams_r'] = ta.WILLR(result_df['High'], result_df['Low'], result_df['Close'], timeperiod=14)

            # 10. Stochastic (2 indicators)
            stoch_k, stoch_d = ta.STOCH(result_df['High'], result_df['Low'], result_df['Close'])
            result_df['stoch_k'] = stoch_k
            result_df['stoch_d'] = stoch_d

            # 11. Price Change (1 indicator)
            result_df['price_change'] = result_df['Close'].pct_change()

            # 12. Volume indicators (2 indicators)
            if 'Volume' in result_df.columns:
                result_df['volume_sma'] = ta.SMA(result_df['Volume'], timeperiod=10)
                result_df['volume_ratio'] = result_df['Volume'] / result_df['volume_sma']
            else:
                result_df['volume_sma'] = 1.0
                result_df['volume_ratio'] = 1.0

            # 13. Additional price-based indicators (3 indicators)
            result_df['high_low_ratio'] = (result_df['High'] - result_df['Low']) / result_df['Close']
            result_df['close_sma_ratio'] = result_df['Close'] / result_df['sma_20']
            result_df['volatility_5'] = result_df['price_change'].rolling(5).std()

            # Total: 23 indicators
            expected_indicators = [
                'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'atr', 'adx', 'cci', 'williams_r',
                'stoch_k', 'stoch_d', 'price_change', 'volume_sma', 'volume_ratio',
                'high_low_ratio', 'close_sma_ratio', 'volatility_5'
            ]

            # Verify all indicators were created
            missing_indicators = [ind for ind in expected_indicators if ind not in result_df.columns]
            if missing_indicators:
                print(f"Warning: Missing indicators: {missing_indicators}")

            # Fill NaN values using forward fill then backward fill
            for indicator in expected_indicators:
                if indicator in result_df.columns:
                    result_df[indicator] = result_df[indicator].fillna(method='ffill').fillna(method='bfill')

            # For any remaining NaN values, fill with reasonable defaults
            numeric_columns = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_columns] = result_df[numeric_columns].fillna(0)

            print(f"✅ Calculated 23 technical indicators successfully")
            return result_df

        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            # Return original dataframe if calculation fails
            return df.copy()

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of all feature names."""
        return [
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_width', 'atr', 'adx', 'cci', 'williams_r',
            'stoch_k', 'stoch_d', 'price_change', 'volume_sma', 'volume_ratio',
            'high_low_ratio', 'close_sma_ratio', 'volatility_5'
        ]