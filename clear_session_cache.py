#!/usr/bin/env python3
"""
Clear all session state cache containing synthetic datetime values
"""

import streamlit as st

def clear_all_session_cache():
    """Clear all cached session state to remove synthetic datetime values"""
    
    # Keys that may contain synthetic datetime data
    cache_keys = [
        'features', 'direction_features', 'profit_prob_features', 'reversal_features',
        'data', 'uploaded_data', 'prices', 'recent_prices',
        'volatility_model', 'direction_model', 'profit_prob_model', 'reversal_model',
        'direction_trained_models', 'profit_trained_models', 'reversal_trained_models',
        'predictions', 'direction_predictions', 'profit_predictions', 'reversal_predictions'
    ]
    
    # Clear all potentially problematic cache
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]
            print(f"✅ Cleared session cache: {key}")
    
    # Clear entire session state to be safe
    st.session_state.clear()
    print("✅ Completely cleared all session state")
    print("✅ All cached synthetic datetime values removed")
    print("✅ Next page load will use fresh database data only")

if __name__ == "__main__":
    clear_all_session_cache()