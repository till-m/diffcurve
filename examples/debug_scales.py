import numpy as np
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping

def debug_scale_progression():
    """Debug the scale progression to identify where the issue occurs"""
    
    print("=== Debugging Scale Progression ===")
    
    img = np.random.rand(64, 64)
    N1, N2 = img.shape
    
    for nbscales in [3, 4, 5, 6]:
        print(f"\n--- Testing {nbscales} scales ---")
        
        try:
            # Forward transform
            coeffs = fdct_wrapping(img, is_real=0, finest=2, nbscales=nbscales, nbangles_coarse=16)
            
            print(f"Number of coefficient arrays: {len(coeffs)}")
            
            # Analyze coefficient structure
            for j, scale in enumerate(coeffs):
                print(f"  Scale {j}: {len(scale)} wedges")
                if len(scale) > 0:
                    shapes = [wedge.shape for wedge in scale]
                    print(f"    Shapes: {shapes[:5]}{'...' if len(shapes) > 5 else ''}")
                    
                    # Check for suspicious (1,1) shapes
                    ones_count = sum(1 for shape in shapes if shape == (1, 1))
                    if ones_count > 0:
                        print(f"    WARNING: {ones_count} wedges have (1,1) shape!")
            
            # Test reconstruction
            reconstructed = ifdct_wrapping(coeffs, is_real=0)
            mse = np.mean(np.abs(img - reconstructed)**2)
            print(f"  Reconstruction MSE: {mse}")
            
            if mse > 1e-10:
                print(f"  ❌ Poor reconstruction quality")
            else:
                print(f"  ✅ Good reconstruction quality")
                
        except Exception as e:
            print(f"  ERROR: {e}")

def debug_m1_m2_progression():
    """Debug the M1, M2 coordinate progression"""
    
    print("\n=== Debugging M1, M2 Coordinate Progression ===")
    
    N1, N2 = 64, 64
    
    for nbscales in [3, 4, 5]:
        print(f"\n--- {nbscales} scales ---")
        
        M1 = N1 / 3
        M2 = N2 / 3
        print(f"Initial: M1={M1:.3f}, M2={M2:.3f}")
        
        # Simulate the finest=2 path
        M1 = M1 / 2
        M2 = M2 / 2
        print(f"After finest=2 division: M1={M1:.3f}, M2={M2:.3f}")
        
        # Simulate the scale loop progression
        scales = list(range(nbscales - 2, 1, -1))
        print(f"Scale loop: {scales}")
        
        for j in scales:
            M1 = M1 / 2
            M2 = M2 / 2
            
            # Calculate derived values
            window_length_1 = int(2*M1) - int(M1) - 1
            window_length_2 = int(2*M2) - int(M2) - 1
            bigN1 = 2 * int(4 * M1) + 1
            bigN2 = 2 * int(4 * M2) + 1
            
            print(f"  Scale {j}: M1={M1:.3f}, M2={M2:.3f}")
            print(f"    window_length: {window_length_1} x {window_length_2}")
            print(f"    bigN: {bigN1} x {bigN2}")
            
            # Check for problematic values
            if window_length_1 <= 0 or window_length_2 <= 0:
                print(f"    ❌ PROBLEM: window_length <= 0")
            if bigN1 <= 0 or bigN2 <= 0:
                print(f"    ❌ PROBLEM: bigN <= 0")
            if M1 < 1 or M2 < 1:
                print(f"    ⚠️  WARNING: M1 or M2 < 1")

def debug_nbangles_calculation():
    """Debug the nbangles calculation"""
    
    print("\n=== Debugging nbangles Calculation ===")
    
    nbangles_coarse = 16
    
    for nbscales in [3, 4, 5, 6]:
        print(f"\n--- {nbscales} scales ---")
        
        # Replicate the nbangles calculation from fdct_wrapping.py
        nbangles = [1]
        for j in range(2, nbscales + 1):
            angles = nbangles_coarse * (2 ** int(np.ceil((nbscales - j + 1) / 2)))
            nbangles.append(angles)
        
        # finest=2 modification
        nbangles[nbscales - 1] = 1
        
        print(f"nbangles: {nbangles}")
        
        # Check for problems
        for j, angles in enumerate(nbangles):
            if angles <= 0:
                print(f"  ❌ PROBLEM: Scale {j} has {angles} angles")
            elif angles > 100:
                print(f"  ⚠️  WARNING: Scale {j} has {angles} angles (very high)")

if __name__ == "__main__":
    debug_scale_progression()
    debug_m1_m2_progression() 
    debug_nbangles_calculation()