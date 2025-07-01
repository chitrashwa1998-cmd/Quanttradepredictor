#!/usr/bin/env python3
"""
Fix datetime display in profit probability and reversal prediction sections
"""

def fix_datetime_sections():
    """Fix the datetime display sections to use real timestamps only"""
    
    # Read the file
    with open('pages/3_Predictions.py', 'r') as f:
        content = f.read()
    
    # Fix profit probability section - replace the entire try-except block
    profit_prob_old = '''                # Create the main predictions dataframe with improved datetime handling
                try:
                    # Use the same successful approach as volatility/direction models
                    if hasattr(recent_prices_aligned.index, 'strftime'):
                        # Already datetime index
                        date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                        time_col = recent_prices_aligned.index.strftime('%H:%M:%S')
                    else:
                        # Use the original data index which should have proper datetime
                        original_index = st.session_state.data.index[-len(recent_prices_aligned):]
                        if hasattr(original_index, 'strftime'):
                            date_col = original_index.strftime('%Y-%m-%d')
                            time_col = original_index.strftime('%H:%M:%S')
                        else:
                            # Fallback: create realistic datetime sequence
                            start_idx = len(st.session_state.data) - len(recent_prices_aligned)
                            date_col = [f"Point_{start_idx + i + 1}" for i in range(len(recent_prices_aligned))]
                            time_col = [f"{(9 + (i % 390) // 12):02d}:{((i % 12) * 5):02d}:00" for i in range(len(recent_prices_aligned))]
                        
                except Exception as e:
                    # Fallback with realistic market time simulation
                    start_idx = len(st.session_state.data) - len(recent_prices_aligned)
                    date_col = [f"Point_{start_idx + i + 1}" for i in range(len(recent_prices_aligned))]
                    time_col = [f"{(9 + (i % 390) // 12):02d}:{((i % 12) * 5):02d}:00" for i in range(len(recent_prices_aligned))]'''
    
    profit_prob_new = '''                # Create the main predictions dataframe using actual timestamps
                date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                time_col = recent_prices_aligned.index.strftime('%H:%M:%S')'''
    
    # Replace profit probability section
    if profit_prob_old in content:
        content = content.replace(profit_prob_old, profit_prob_new)
        print("✅ Fixed profit probability datetime section")
    else:
        print("❌ Profit probability section not found")
    
    # Fix reversal section - find similar pattern
    reversal_old = '''                # Create reversal predictions dataframe with improved datetime handling
                try:
                    # Use the same successful approach as volatility/direction models
                    if hasattr(recent_prices_aligned.index, 'strftime'):
                        # Already datetime index
                        date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                        time_col = recent_prices_aligned.index.strftime('%H:%M:%S')
                    else:
                        # Use the original data index which should have proper datetime
                        original_index = st.session_state.data.index[-len(recent_prices_aligned):]
                        if hasattr(original_index, 'strftime'):
                            date_col = original_index.strftime('%Y-%m-%d')
                            time_col = original_index.strftime('%H:%M:%S')
                        else:
                            # Fallback: create realistic datetime sequence
                            start_idx = len(st.session_state.data) - len(recent_prices_aligned)
                            date_col = [f"Point_{start_idx + i + 1}" for i in range(len(recent_prices_aligned))]
                            time_col = [f"{(9 + (i % 390) // 12):02d}:{((i % 12) * 5):02d}:00" for i in range(len(recent_prices_aligned))]
                        
                except Exception as e:
                    # Fallback with realistic market time simulation
                    start_idx = len(st.session_state.data) - len(recent_prices_aligned)
                    date_col = [f"Point_{start_idx + i + 1}" for i in range(len(recent_prices_aligned))]
                    time_col = [f"{(9 + (i % 390) // 12):02d}:{((i % 12) * 5):02d}:00" for i in range(len(recent_prices_aligned))]'''
    
    reversal_new = '''                # Create reversal predictions dataframe using actual timestamps
                date_col = recent_prices_aligned.index.strftime('%Y-%m-%d')
                time_col = recent_prices_aligned.index.strftime('%H:%M:%S')'''
    
    # Replace reversal section
    if reversal_old in content:
        content = content.replace(reversal_old, reversal_new)
        print("✅ Fixed reversal datetime section")
    else:
        print("❌ Reversal section not found")
    
    # Write the file back
    with open('pages/3_Predictions.py', 'w') as f:
        f.write(content)
    
    print("✅ Datetime fixes completed")

if __name__ == "__main__":
    fix_datetime_sections()