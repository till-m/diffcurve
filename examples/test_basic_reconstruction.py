import numpy as np
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping

def test_basic_reconstruction():
    """Test the most basic reconstruction: forward -> inverse should give back the original"""
    
    print("=== Basic Reconstruction Test ===")
    
    # Create simple test image
    img = np.random.rand(64, 64)
    print(f"Original image shape: {img.shape}")
    print(f"Original image range: [{img.min():.4f}, {img.max():.4f}]")
    
    # Forward transform
    print("Running forward transform...")
    coeffs = fdct_wrapping(img, is_real=0, finest=2, nbscales=4, nbangles_coarse=16)
    
    # Print coefficient structure
    print(f"Number of scales: {len(coeffs)}")
    for j, scale in enumerate(coeffs):
        print(f"  Scale {j}: {len(scale)} wedges")
        for l, wedge in enumerate(scale):
            print(f"    Wedge {l}: shape {wedge.shape}, type {wedge.dtype}")
    
    # Inverse transform
    print("Running inverse transform...")
    reconstructed = ifdct_wrapping(coeffs, is_real=0)
    
    print(f"Reconstructed image shape: {reconstructed.shape}")
    print(f"Reconstructed image range: [{reconstructed.real.min():.4f}, {reconstructed.real.max():.4f}]")
    
    # Calculate errors
    mse = np.mean(np.abs(img - reconstructed)**2)
    max_error = np.max(np.abs(img - reconstructed))
    
    print(f"MSE: {mse}")
    print(f"Max absolute error: {max_error}")
    print(f"Relative MSE: {mse / np.mean(img**2)}")
    
    # Test if the issue is in the complex part
    if np.iscomplexobj(reconstructed):
        mse_real = np.mean((img - reconstructed.real)**2)
        imag_magnitude = np.mean(np.abs(reconstructed.imag)**2)
        print(f"MSE (real part only): {mse_real}")
        print(f"Imaginary part magnitude: {imag_magnitude}")
    
    return mse

def test_different_parameters():
    """Test with different parameter combinations"""
    
    print("\n=== Parameter Variation Test ===")
    
    img = np.random.rand(32, 32)
    
    test_configs = [
        {"is_real": 0, "finest": 2, "nbscales": 3, "nbangles_coarse": 8},
        {"is_real": 0, "finest": 2, "nbscales": 4, "nbangles_coarse": 16},
        {"is_real": 1, "finest": 2, "nbscales": 3, "nbangles_coarse": 8},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        try:
            coeffs = fdct_wrapping(img, **config)
            reconstructed = ifdct_wrapping(coeffs, is_real=config["is_real"])
            
            if config["is_real"]:
                mse = np.mean((img - reconstructed)**2)
            else:
                mse = np.mean(np.abs(img - reconstructed)**2)
            
            print(f"  MSE: {mse}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    mse = test_basic_reconstruction()
    test_different_parameters()
    
    print(f"\n=== Summary ===")
    print(f"Primary MSE: {mse}")
    if mse > 1e-10:
        print("❌ High reconstruction error - implementation has issues")
    else:
        print("✅ Good reconstruction quality")