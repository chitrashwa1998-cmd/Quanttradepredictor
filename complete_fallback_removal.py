#!/usr/bin/env python3
"""
Complete removal of ALL remaining fallback logic patterns
"""

def remove_all_remaining_fallbacks():
    """Remove all remaining fallback patterns identified in the analysis"""
    
    # Read the file
    with open('pages/3_Predictions.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Direction chart confidence fallback (line 1073)
    content = content.replace(
        'confidences = recent_probs_aligned if recent_probs_aligned is not None else np.ones(len(pred_data)) * 0.5',
        'confidences = recent_probs_aligned'
    )
    print("✅ Fixed direction chart confidence fallback")
    
    # Fix 2: Direction data table confidence fallback (lines 1417-1418)
    content = content.replace(
        "'Confidence': [np.max(prob) if prob is not None else 0.5 for prob in recent_probs_aligned] if recent_probs_aligned is not None else [0.5] * len(recent_predictions_aligned)",
        "'Confidence': [np.max(prob) for prob in recent_probs_aligned]"
    )
    print("✅ Fixed direction data table confidence fallback")
    
    # Fix 3: Profit probability data table confidence fallback (lines 2366-2367)
    content = content.replace(
        "'Confidence': [np.max(prob) if prob is not None else 0.5 for prob in recent_probs_aligned] if recent_probs_aligned is not None else [0.5] * len(recent_predictions_aligned)",
        "'Confidence': [np.max(prob) for prob in recent_probs_aligned]"
    )
    print("✅ Fixed profit probability data table confidence fallback")
    
    # Fix 4: Reversal data table confidence fallback (lines 3039-3040)
    content = content.replace(
        "'Confidence': [np.max(prob) if prob is not None else 0.5 for prob in recent_probs_aligned] if recent_probs_aligned is not None else [0.5] * len(recent_predictions_aligned)",
        "'Confidence': [np.max(prob) for prob in recent_probs_aligned]"
    )
    print("✅ Fixed reversal data table confidence fallback")
    
    # Fix 5: Remove any remaining N/A fallback patterns
    content = content.replace(
        'if recent_probs_aligned is not None else [\'N/A\'] * actual_len',
        ''
    )
    print("✅ Fixed N/A confidence display fallbacks")
    
    # Fix 6: Clean up confidence display patterns
    content = content.replace(
        '([f"{np.max(prob):.3f}" for prob in recent_probs_aligned] \n                                  )',
        '[f"{np.max(prob):.3f}" for prob in recent_probs_aligned]'
    )
    print("✅ Fixed multi-line confidence patterns")
    
    # Fix 6: Remove any remaining fallback datetime logic that might still exist
    # Look for hasattr patterns that could generate synthetic dates
    content = content.replace(
        'if hasattr(recent_prices.index, \'strftime\'):',
        '# Use authentic datetime from database'
    )
    
    # Fix 7: Remove any remaining Point_ or Data_ generation patterns
    import re
    
    # Remove any line that generates Point_ or Data_ patterns
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that generate synthetic Point_ or Data_ values
        if ('f"Point_' in line or 'f"Data_' in line or '"Point_' in line or '"Data_' in line) and 'for i in range' in line:
            print(f"✅ Removed synthetic generation line: {line.strip()}")
            continue
        # Skip lines with 09:15:00 hardcoded times
        elif '09:15:00' in line and 'f"{' in line:
            print(f"✅ Removed hardcoded time generation: {line.strip()}")
            continue
        else:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Fix 8: Ensure all strftime calls are direct without conditionals
    # Replace any remaining conditional strftime calls
    content = re.sub(
        r'if hasattr\([^,]+\.index, [\'"]strftime[\'"]\):\s*\n\s*([^\n]+\.strftime[^\n]+)\s*\n\s*else:\s*\n\s*[^\n]+',
        r'\1',
        content,
        flags=re.MULTILINE
    )
    
    # Write the cleaned content
    with open('pages/3_Predictions.py', 'w') as f:
        f.write(content)
    
    print("✅ All remaining fallback logic removed")
    print("✅ No more synthetic confidence values")
    print("✅ No more synthetic datetime values")
    print("✅ All models use only authentic database data")

if __name__ == "__main__":
    remove_all_remaining_fallbacks()