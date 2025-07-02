#!/usr/bin/env python3
"""
Performance benchmark for optimized curvelet transform.
"""

import time
import numpy as np
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping


def benchmark_curvelet_transform(image_size=128, num_runs=5):
    """Benchmark the curvelet transform performance."""
    print(f"Benchmarking curvelet transform on {image_size}x{image_size} image")
    print(f"Number of runs: {num_runs}")
    print("-" * 50)
    
    # Create test image
    np.random.seed(42)  # For reproducible results
    img = np.random.randn(image_size, image_size)
    
    forward_times = []
    inverse_times = []
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Benchmark forward transform
        start_time = time.perf_counter()
        coeffs = fdct_wrapping(img)
        forward_time = time.perf_counter() - start_time
        forward_times.append(forward_time)
        
        # Benchmark inverse transform
        start_time = time.perf_counter()
        reconstructed = ifdct_wrapping(coeffs)
        inverse_time = time.perf_counter() - start_time
        inverse_times.append(inverse_time)
        
        # Verify reconstruction quality
        mse = np.mean(np.abs(img - reconstructed)**2)
        print(f"  Forward: {forward_time:.4f}s, Inverse: {inverse_time:.4f}s, MSE: {mse:.2e}")
    
    print("-" * 50)
    print("Performance Summary:")
    print(f"Forward transform:")
    print(f"  Mean: {np.mean(forward_times):.4f}s ± {np.std(forward_times):.4f}s")
    print(f"  Min:  {np.min(forward_times):.4f}s")
    print(f"  Max:  {np.max(forward_times):.4f}s")
    
    print(f"Inverse transform:")
    print(f"  Mean: {np.mean(inverse_times):.4f}s ± {np.std(inverse_times):.4f}s")
    print(f"  Min:  {np.min(inverse_times):.4f}s")
    print(f"  Max:  {np.max(inverse_times):.4f}s")
    
    print(f"Total transform time: {np.mean(forward_times) + np.mean(inverse_times):.4f}s")
    
    return {
        'forward_times': forward_times,
        'inverse_times': inverse_times,
        'forward_mean': np.mean(forward_times),
        'inverse_mean': np.mean(inverse_times),
        'total_mean': np.mean(forward_times) + np.mean(inverse_times)
    }


if __name__ == "__main__":
    # Benchmark different image sizes
    sizes = [64, 128]
    
    for size in sizes:
        print(f"\n{'='*60}")
        benchmark_curvelet_transform(size, num_runs=3)
        print(f"{'='*60}")