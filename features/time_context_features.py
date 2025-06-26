import pandas as pd
import numpy as np

def create_time_context_features(df):
    """
    No time context features - using only the 5 specified technical indicators:
    ATR, Bollinger Band Width, Keltner Channel Width, RSI, Donchian Channel Width
    """
    # Return dataframe unchanged - all features come from technical_indicators.py
    return df