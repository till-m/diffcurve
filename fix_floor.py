#!/usr/bin/env python3
"""Fix int() calls to use np.floor() to match MATLAB behavior"""

import re

def fix_floor_calls(file_path):
    """Replace int(N*M...) with int(np.floor(N*M...)) patterns"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Patterns to replace
    patterns = [
        (r'int\(4\*M_horiz\)', r'int(np.floor(4*M_horiz))'),
        (r'int\(4\*M_vert\)', r'int(np.floor(4*M_vert))'),
        (r'int\(2\*M1\)', r'int(np.floor(2*M1))'),
        (r'int\(2\*M2\)', r'int(np.floor(2*M2))'),
        (r'int\(M1\)', r'int(np.floor(M1))'),
        (r'int\(M2\)', r'int(np.floor(M2))'),
        (r'int\(M_vert\)', r'int(np.floor(M_vert))'),
        (r'int\(M_horiz\)', r'int(np.floor(M_horiz))'),
        # More specific patterns for complex expressions
        (r'2\*int\(4\*M_horiz\)', r'2*int(np.floor(4*M_horiz))'),
        (r'2\*int\(4\*M_vert\)', r'2*int(np.floor(4*M_vert))'),
        (r'4\*int\(4\*M_horiz\)', r'4*int(np.floor(4*M_horiz))'),
    ]
    
    # Apply replacements
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed floor calls in {file_path}")

if __name__ == "__main__":
    fix_floor_calls('/Users/muser0001/diffcurve/diffcurve/fdct2d/fdct_wrapping.py')
    fix_floor_calls('/Users/muser0001/diffcurve/diffcurve/fdct2d/ifdct_wrapping.py')
    print("Done!")