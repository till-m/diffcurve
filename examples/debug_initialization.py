import numpy as np
from diffcurve.fdct2d import fdct_wrapping

def debug_array_sizes():
    """Debug array size calculations"""
    
    print("=== Debugging Array Size Calculations ===")
    
    # Test with the problematic 512x512 case
    N1, N2 = 512, 512
    nbscales = 6
    
    print(f"Image size: {N1} x {N2}")
    print(f"nbscales: {nbscales}")
    
    # Initial M1, M2
    M1 = N1 / 3
    M2 = N2 / 3
    print(f"Initial M1, M2: {M1:.3f}, {M2:.3f}")
    
    # finest=2 path
    M1 = M1 / 2
    M2 = M2 / 2
    print(f"After finest=2: M1, M2: {M1:.3f}, {M2:.3f}")
    
    # Calculate window lengths and bigN arrays for the initial step
    window_length_1 = int(np.floor(2*M1)) - int(np.floor(M1)) - 1
    window_length_2 = int(np.floor(2*M2)) - int(np.floor(M2)) - 1
    
    print(f"Initial window_length: {window_length_1} x {window_length_2}")
    
    # Calculate the coordinate arrays
    coord_1 = np.linspace(0, 1, window_length_1 + 1)
    coord_2 = np.linspace(0, 1, window_length_2 + 1)
    
    print(f"Coordinate array lengths: {len(coord_1)} x {len(coord_2)}")
    
    # Calculate lowpass sizes
    lowpass_1_size = len(coord_1[:-1]) + (2*int(np.floor(M1))+1) + len(coord_1[:-1])
    lowpass_2_size = len(coord_2[:-1]) + (2*int(np.floor(M2))+1) + len(coord_2[:-1])
    
    print(f"Expected lowpass sizes: {lowpass_1_size} x {lowpass_2_size}")
    
    # Now check what the actual forward transform produces
    print("\n--- Actual Forward Transform Results ---")
    
    img = np.random.rand(N1, N2)
    coeffs = fdct_wrapping(img, is_real=0, finest=2, nbscales=nbscales, nbangles_coarse=16)
    
    print(f"Generated {len(coeffs)} scales")
    
    for j, scale in enumerate(coeffs):
        print(f"Scale {j}: {len(scale)} wedges")
        if len(scale) > 0:
            total_elements = sum(coeff.size for coeff in scale)
            example_shape = scale[0].shape if len(scale) > 0 else "None"
            print(f"  Example shape: {example_shape}")
            print(f"  Total elements in scale: {total_elements}")

def check_matlab_vs_python_differences():
    """Check for specific differences that could cause issues"""
    
    print("\n=== Checking MATLAB vs Python Differences ===")
    
    # Test some specific calculations
    M1, M2 = 85.333, 85.333  # From our earlier debug
    
    print(f"M1, M2 = {M1}, {M2}")
    
    # Compare floor vs int
    print(f"int(2*M1) = {int(2*M1)}")
    print(f"int(np.floor(2*M1)) = {int(np.floor(2*M1))}")
    print(f"int(M1) = {int(M1)}")  
    print(f"int(np.floor(M1)) = {int(np.floor(M1))}")
    
    # Window length calculation
    window_length_matlab = int(np.floor(2*M1)) - int(np.floor(M1)) - 1
    window_length_python_old = int(2*M1) - int(M1) - 1
    
    print(f"Window length (MATLAB style): {window_length_matlab}")
    print(f"Window length (old Python): {window_length_python_old}")
    
    # Check coordinate spacing
    if window_length_matlab > 0:
        coord_matlab = np.linspace(0, 1, window_length_matlab + 1)
        print(f"Coordinate array length: {len(coord_matlab)}")
        print(f"Coordinate spacing: {1.0/window_length_matlab if window_length_matlab > 0 else 'N/A'}")

if __name__ == "__main__":
    debug_array_sizes()
    check_matlab_vs_python_differences()