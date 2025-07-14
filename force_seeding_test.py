#!/usr/bin/env python3
"""
Force seeding test to verify the seeding works correctly
"""

import streamlit as st
import pandas as pd
from utils.live_data_manager import LiveDataManager
from utils.database_adapter import DatabaseAdapter

def force_seeding_test():
    """Test forced seeding with actual data"""
    print("🔍 Testing forced seeding...")
    
    # Initialize the live data manager
    manager = LiveDataManager("dummy_token", "dummy_key")
    
    # Force seeding
    result = manager.seed_live_data_from_database("NSE_INDEX|Nifty 50")
    
    if result:
        print("✅ Seeding successful!")
        print(f"Seeded instruments: {manager.seeded_instruments}")
        
        if "NSE_INDEX|Nifty 50" in manager.ohlc_data:
            data = manager.ohlc_data["NSE_INDEX|Nifty 50"]
            print(f"OHLC data shape: {data.shape}")
            print(f"OHLC data columns: {list(data.columns)}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            # Save the seeded data to session state for the live system
            if 'live_data_manager' not in st.session_state:
                st.session_state.live_data_manager = manager
                print("✅ Saved seeded data to session state")
            else:
                # Update existing manager
                st.session_state.live_data_manager.ohlc_data["NSE_INDEX|Nifty 50"] = data
                st.session_state.live_data_manager.seeded_instruments["NSE_INDEX|Nifty 50"] = manager.seeded_instruments["NSE_INDEX|Nifty 50"]
                print("✅ Updated existing manager with seeded data")
        else:
            print("❌ No OHLC data found after seeding")
    else:
        print("❌ Seeding failed")

# Run the test
if __name__ == "__main__":
    print("Running force seeding test...")
    force_seeding_test()