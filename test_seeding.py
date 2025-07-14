#!/usr/bin/env python3
"""
Test script to check if seeding function works correctly
"""

from utils.live_data_manager import LiveDataManager
from utils.database_adapter import DatabaseAdapter
import os

def test_seeding():
    """Test the seeding function directly."""
    print("Testing seeding function...")
    
    # Create a test manager
    manager = LiveDataManager("test_token", "test_key")
    
    # Test the seeding function
    result = manager.seed_live_data_from_database("NSE_INDEX|Nifty 50")
    
    print(f"Seeding result: {result}")
    
    # Check if data was seeded
    if "NSE_INDEX|Nifty 50" in manager.seeded_instruments:
        print(f"Seeded data info: {manager.seeded_instruments['NSE_INDEX|Nifty 50']}")
    else:
        print("No seeded data found")
        
    # Check database directly
    print("\nDirect database check:")
    db = DatabaseAdapter()
    try:
        data = db.load_ohlc_data("main_dataset")
        if data is not None:
            print(f"Found {len(data)} rows in main_dataset")
            print(f"Columns: {list(data.columns)}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
        else:
            print("No data found in main_dataset")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    test_seeding()