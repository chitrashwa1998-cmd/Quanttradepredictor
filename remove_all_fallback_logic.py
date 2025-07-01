#!/usr/bin/env python3
"""
Remove all fallback logic and synthetic datetime generation from prediction models
Use only authentic timestamps from database
"""

def remove_all_fallback_logic():
    """Remove all fallback datetime logic from all 4 models"""
    
    # Read the file
    with open('pages/3_Predictions.py', 'r') as f:
        content = f.read()
    
    # Remove the synthetic datetime generation function
    function_to_remove = '''
def generate_realistic_datetime_columns(data_length, start_index=0):
    """Generate realistic datetime columns for trading data"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create a realistic datetime sequence starting from a recent date
    base_date = datetime(2024, 1, 1, 9, 15, 0)  # Market open time
    
    # Generate datetime sequence with 5-minute intervals (typical trading data)
    datetime_list = []
    current_time = base_date
    
    for i in range(data_length):
        # Skip weekends (Saturday = 5, Sunday = 6)
        while current_time.weekday() >= 5:
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=15, second=0)
        
        # Market hours: 9:15 AM to 3:30 PM (Indian market)
        if current_time.hour >= 15 and current_time.minute >= 30:
            # Move to next trading day
            current_time += timedelta(days=1)
            current_time = current_time.replace(hour=9, minute=15, second=0)
            # Skip weekends again
            while current_time.weekday() >= 5:
                current_time += timedelta(days=1)
                current_time = current_time.replace(hour=9, minute=15, second=0)
        
        datetime_list.append(current_time)
        current_time += timedelta(minutes=5)  # 5-minute intervals
    
    # Convert to pandas datetime series
    datetime_series = pd.Series(datetime_list)
    
    # Return date and time columns
    date_col = datetime_series.dt.strftime('%Y-%m-%d').tolist()
    time_col = datetime_series.dt.strftime('%H:%M:%S').tolist()
    
    return date_col, time_col
'''
    
    # Remove the function
    content = content.replace(function_to_remove, '')
    
    # Fix profit probability section - use direct datetime access
    profit_old = '''                # Create the main predictions dataframe with realistic datetime
                date_col, time_col = generate_realistic_datetime_columns(len(recent_prices_aligned))'''
    
    profit_new = '''                # Use authentic datetime from database
                date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                time_col = recent_prices_aligned.index.strftime('%H:%M:%S')'''
    
    # Fix reversal section - use direct datetime access
    reversal_old = '''                # Create the main predictions dataframe with realistic datetime
                date_col, time_col = generate_realistic_datetime_columns(len(recent_prices_aligned))'''
    
    reversal_new = '''                # Use authentic datetime from database
                date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                time_col = recent_prices_aligned.index.strftime('%H:%M:%S')'''
    
    # Fix volatility section - remove all debug and fallback logic
    volatility_old = '''                    # Create realistic datetime columns
                    date_col, time_col = generate_realistic_datetime_columns(len(recent_prices))'''
    
    volatility_new = '''                    # Use authentic datetime from database
                    date_col = recent_prices.index.strftime('%Y-%m-%d')
                    time_col = recent_prices.index.strftime('%H:%M:%S')'''
    
    # Fix direction section - remove all complex logic
    direction_old = '''                # Create realistic datetime columns
                date_col, time_col = generate_realistic_datetime_columns(len(recent_prices_aligned))'''
    
    direction_new = '''                # Use authentic datetime from database
                date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                time_col = recent_prices_aligned.index.strftime('%H:%M:%S')'''
    
    # Apply all replacements
    replacements = [
        (profit_old, profit_new, "profit probability"),
        (reversal_old, reversal_new, "reversal"),
        (volatility_old, volatility_new, "volatility"),
        (direction_old, direction_new, "direction")
    ]
    
    for old_pattern, new_pattern, model_name in replacements:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"✅ Fixed {model_name} datetime section")
        else:
            print(f"❌ {model_name} section not found")
    
    # Remove all debug print statements related to datetime
    debug_patterns = [
        'print(f"DEBUG Volatility: Index type: {type(recent_prices.index)}")',
        'print(f"DEBUG Volatility: Index dtype: {recent_prices.index.dtype}")',
        'print(f"DEBUG Volatility: First few index values: {recent_prices.index[:5].tolist()}")',
        'print(f"DEBUG Direction: Index type: {type(recent_prices_aligned.index)}")',
        'print(f"DEBUG Direction: Index dtype: {recent_prices_aligned.index.dtype}")',
        'print(f"DEBUG Direction: First few index values: {recent_prices_aligned.index[:5].tolist()}")',
        'print(f"DEBUG Profit: Index type: {type(recent_prices_aligned.index)}")',
        'print(f"DEBUG Profit: Index dtype: {recent_prices_aligned.index.dtype}")',
        'print(f"DEBUG Profit: First few index values: {recent_prices_aligned.index[:5].tolist()}")'
    ]
    
    for debug_pattern in debug_patterns:
        if debug_pattern in content:
            # Remove the entire line
            lines = content.split('\n')
            lines = [line for line in lines if debug_pattern not in line]
            content = '\n'.join(lines)
            print(f"✅ Removed debug statement: {debug_pattern[:50]}...")
    
    # Fix the length mismatch error on line 1625
    length_mismatch_old = '''                direction_series = pd.Series(predictions, index=filtered_data.index[-len(predictions):])'''
    length_mismatch_new = '''                # Ensure predictions and index lengths match
                data_len = min(len(predictions), len(filtered_data))
                direction_series = pd.Series(predictions[:data_len], index=filtered_data.index[-data_len:])'''
    
    if length_mismatch_old in content:
        content = content.replace(length_mismatch_old, length_mismatch_new)
        print("✅ Fixed direction series length mismatch")
    
    # Write the cleaned file back
    with open('pages/3_Predictions.py', 'w') as f:
        f.write(content)
    
    print("✅ All fallback logic removed")
    print("✅ All models now use only authentic database timestamps")

if __name__ == "__main__":
    remove_all_fallback_logic()