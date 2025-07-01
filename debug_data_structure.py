#!/usr/bin/env python3
"""
Debug script to check the actual data structure and create proper datetime handling
"""
import streamlit as st
import pandas as pd
from utils.database_adapter import get_trading_database

def debug_data_structure():
    """Debug the actual data structure"""
    try:
        # Load data from database
        db = get_trading_database()
        data = db.recover_data()
        
        if data is not None and not data.empty:
            print("=== DATA STRUCTURE DEBUG ===")
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {list(data.columns)}")
            print(f"Index type: {type(data.index)}")
            print(f"Index dtype: {data.index.dtype}")
            print(f"First 10 index values: {data.index[:10].tolist()}")
            print(f"Last 10 index values: {data.index[-10:].tolist()}")
            
            # Check if there are datetime-like columns
            print("\n=== COLUMN ANALYSIS ===")
            for col in data.columns:
                print(f"Column '{col}': dtype={data[col].dtype}")
                if col.lower() in ['date', 'time', 'datetime', 'timestamp']:
                    print(f"  Sample values: {data[col].head().tolist()}")
                    
            # Check first few rows
            print("\n=== FIRST 5 ROWS ===")
            print(data.head())
            
            # Check for any datetime patterns in the data
            print("\n=== LOOKING FOR DATETIME PATTERNS ===")
            sample_rows = data.head(10)
            for col in sample_rows.columns:
                sample_vals = sample_rows[col].dropna()
                if len(sample_vals) > 0:
                    first_val = sample_vals.iloc[0]
                    if isinstance(first_val, (int, float)):
                        if first_val > 1e9:  # Could be timestamp
                            print(f"Column '{col}' might be timestamp - first value: {first_val}")
                    elif isinstance(first_val, str):
                        if any(sep in str(first_val) for sep in ['/', '-', ':']):
                            print(f"Column '{col}' might be datetime string - first value: {first_val}")
            
            print("\n=== RECOMMENDATIONS ===")
            print("Based on the structure above, we need to:")
            print("1. Identify if there are datetime columns in the data")
            print("2. Create proper datetime index if possible")
            print("3. Generate realistic datetime values if no datetime data exists")
            
        else:
            print("No data found in database")
            
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_data_structure()