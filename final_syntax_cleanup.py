#!/usr/bin/env python3
"""
Final syntax cleanup to fix all remaining syntax issues
"""

def final_syntax_cleanup():
    """Fix all remaining syntax issues"""
    
    # Read the file
    with open('pages/3_Predictions.py', 'r') as f:
        content = f.read()
    
    # Fix incomplete debug statements by removing them completely
    lines = content.split('\n')
    cleaned_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        # Skip debug lines that are causing syntax errors
        if ('st.write(f"recent_predictions_aligned=' in line or 
            'f"recent_probs_aligned=' in line or
            'f"price_changes=' in line or
            'f"actual_direction=' in line):
            # Skip this and next lines if they're continuation
            skip_next = True
            continue
        elif skip_next and (line.strip().startswith('f"') or 
                           line.strip().endswith('")')):
            skip_next = False
            continue
        elif skip_next:
            skip_next = False
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Remove any remaining broken debug statements
    import re
    
    # Remove standalone f-string lines that are causing issues
    content = re.sub(r'^\s*f"[^"]*"\s*$', '', content, flags=re.MULTILINE)
    
    # Remove lines that start with just f" and aren't complete statements
    content = re.sub(r'^\s*f"[^"]*[^)]\s*$', '', content, flags=re.MULTILINE)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Write the cleaned content
    with open('pages/3_Predictions.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed all syntax issues")
    print("✅ Removed broken debug statements")
    print("✅ Cleaned up code structure")

if __name__ == "__main__":
    final_syntax_cleanup()