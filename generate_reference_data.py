#!/usr/bin/env python3
"""
Generate reference data using MATLAB implementation for regression testing.

This script should be run with the original MATLAB code (commit 8e259ac)
to generate ground truth data for testing the optimized Python implementation.
"""

import numpy as np
import matlab.engine
from pathlib import Path
import os


def setup_matlab_engine():
    """Start MATLAB engine and add necessary paths."""
    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    
    # Add the MATLAB fdct2d directory to path
    project_root = Path(__file__).parent
    matlab_fdct_dir = project_root / "matlabfdct2d"
    
    if matlab_fdct_dir.exists():
        eng.addpath(str(matlab_fdct_dir))
        print(f"Added MATLAB path: {matlab_fdct_dir}")
    else:
        print(f"Warning: MATLAB directory not found at {matlab_fdct_dir}")
    
    return eng


def generate_test_case(eng, name, img, **kwargs):
    """Generate a single test case using MATLAB."""
    print(f"Generating test case: {name}")
    
    # Convert to MATLAB format
    img_matlab = matlab.double(img.tolist())
    
    # Set default parameters
    is_real = kwargs.get('is_real', 0)
    finest = kwargs.get('finest', 2)
    
    try:
        # Forward transform
        print(f"  Running MATLAB fdct_wrapping...")
        C = eng.fdct_wrapping(img_matlab, is_real, finest)
        
        # Inverse transform  
        print(f"  Running MATLAB ifdct_wrapping...")
        reconstructed_matlab = eng.ifdct_wrapping(C, is_real)
        
        # Convert back to numpy
        reconstructed = np.array(reconstructed_matlab)
        
        # Calculate MSE
        mse = np.mean(np.abs(img - reconstructed)**2)
        print(f"  MATLAB MSE: {mse:.2e}")
        
        return {
            'image': img,
            'reconstructed': reconstructed,
            'mse': mse,
            'parameters': kwargs
        }
        
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    """Generate comprehensive reference dataset."""
    # Create output directory
    output_dir = Path("tests/reference_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING MATLAB REFERENCE DATA")
    print("="*60)
    
    # Start MATLAB
    eng = setup_matlab_engine()
    
    # Test cases to generate
    test_cases = [
        # Basic sizes with random images
        {
            'name': 'random_32x32',
            'image': np.random.RandomState(42).randn(32, 32),
            'params': {}
        },
        {
            'name': 'random_64x64', 
            'image': np.random.RandomState(123).randn(64, 64),
            'params': {}
        },
        {
            'name': 'random_128x128',
            'image': np.random.RandomState(456).randn(128, 128), 
            'params': {}
        },
        {
            'name': 'random_256x256',
            'image': np.random.RandomState(789).randn(256, 256),
            'params': {}
        },
        
        # Different parameters
        {
            'name': 'real_transform_64x64',
            'image': np.random.RandomState(111).randn(64, 64),
            'params': {'is_real': 1}
        },
        {
            'name': 'finest1_64x64',
            'image': np.random.RandomState(222).randn(64, 64),
            'params': {'finest': 1}
        },
        {
            'name': 'real_finest1_64x64',
            'image': np.random.RandomState(333).randn(64, 64),
            'params': {'is_real': 1, 'finest': 1}
        },
        
        # Non-square images
        {
            'name': 'random_64x128',
            'image': np.random.RandomState(444).randn(64, 128),
            'params': {}
        },
        {
            'name': 'random_128x64',
            'image': np.random.RandomState(555).randn(128, 64),
            'params': {}
        },
        
        # Special test images
        {
            'name': 'constant_64x64',
            'image': np.ones((64, 64)) * 5.0,
            'params': {}
        },
        {
            'name': 'delta_64x64',
            'image': lambda: (lambda img: (img.__setitem__((32, 32), 1.0), img)[1])(np.zeros((64, 64))),
            'params': {}
        },
        {
            'name': 'checkerboard_64x64',
            'image': lambda: ((np.arange(64)[:, None] + np.arange(64)) % 2).astype(float),
            'params': {}
        }
    ]
    
    # Generate each test case
    results = {}
    for case in test_cases:
        name = case['name']
        
        # Handle lambda functions for special images
        if callable(case['image']):
            img = case['image']()
        else:
            img = case['image']
        
        result = generate_test_case(eng, name, img, **case['params'])
        
        if result is not None:
            # Save to file
            output_file = output_dir / f"matlab_{name}.npz"
            np.savez_compressed(output_file, **result)
            print(f"  Saved: {output_file}")
            results[name] = result
        else:
            print(f"  Failed to generate: {name}")
    
    # Stop MATLAB
    eng.quit()
    
    # Summary
    print("\n" + "="*60)
    print("REFERENCE DATA GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(results)} test cases")
    print(f"Saved to: {output_dir}")
    
    # Print MSE summary
    print("\nMSE Summary:")
    for name, result in results.items():
        print(f"  {name:25}: {result['mse']:.2e}")
    
    print(f"\nTotal files: {len(list(output_dir.glob('*.npz')))}")
    print("Ready for regression testing!")


if __name__ == "__main__":
    main()