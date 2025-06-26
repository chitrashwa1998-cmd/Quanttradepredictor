import pandas as pd
import numpy as np

def compute_custom_volatility_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute custom engineered volatility features from OHLC data.
    Assumes df has columns: ['open', 'high', 'low', 'close'] with datetime index.
    Volume is not required.
    """
    df = df.copy()

    # 1. Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Realized volatility (rolling std of log returns)
    df['realized_volatility'] = df['log_return'].rolling(window=window).std()

    # 3. Parkinson volatility
    df['parkinson_volatility'] = (1 / (4 * np.log(2))) * ((np.log(df['high'] / df['low'])) ** 2)
    df['parkinson_volatility'] = df['parkinson_volatility'].rolling(window=window).mean()

    # 4. High-Low ratio
    df['high_low_ratio'] = (df['high'] / df['low']) - 1

    # 5. Gap percentage from previous close
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']

    # 6. Price vs VWAP (using rolling VWAP without volume)
    df['vwap_like'] = (df['high'] + df['low'] + df['close']) / 3
    df['rolling_vwap'] = df['vwap_like'].rolling(window=window).mean()
    df['price_vs_vwap'] = (df['close'] - df['rolling_vwap']) / df['rolling_vwap']

    # 7. Volatility spike flag (candle range exceeds rolling high range by 2 std devs)
    df['candle_range'] = df['high'] - df['low']
    df['range_mean'] = df['candle_range'].rolling(window).mean()
    df['range_std'] = df['candle_range'].rolling(window).std()
    df['volatility_spike_flag'] = (df['candle_range'] > df['range_mean'] + 2 * df['range_std']).astype(int)

    # 8. Candle body asymmetry ratio
    df['body'] = abs(df['close'] - df['open'])
    df['candle_asymmetry_ratio'] = df['body'] / df['candle_range']

    # Optional: drop helper columns
    df.drop(['prev_close', 'vwap_like', 'rolling_vwap', 'range_mean', 'range_std', 'body'], axis=1, inplace=True)

    return df
