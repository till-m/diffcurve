import numpy as np
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping

# Test simple reconstruction without complex decomposition logic
print('Testing simple forward/inverse transform...')

# Create test image
img = np.random.rand(64, 64)

# Forward transform
print('Running forward transform...')
coeffs = fdct_wrapping(img, is_real=0, finest=2, nbscales=6, nbangles_coarse=16)

# Inverse transform
print('Running inverse transform...')
reconstructed = ifdct_wrapping(coeffs, is_real=0)

# Calculate MSE
mse = np.mean((img - reconstructed)**2)
print(f'Reconstruction MSE: {mse}')

# Test with real image data
print('\nTesting with actual image data...')
try:
    import cv2
    from pathlib import Path
    from diffcurve.utils import get_project_root
    
    project_root = get_project_root()
    lena_file = Path.joinpath(project_root, "data/Lena.jpg")
    
    if lena_file.exists():
        lena_img = cv2.imread(str(lena_file), 0).astype(float) / 255
        print(f'Image shape: {lena_img.shape}')
        
        # Forward transform
        lena_coeffs = fdct_wrapping(lena_img, is_real=0, finest=2, nbscales=6, nbangles_coarse=16)
        
        # Inverse transform
        lena_reconstructed = ifdct_wrapping(lena_coeffs, is_real=0)
        
        # Calculate MSE
        lena_mse = np.mean((lena_img - lena_reconstructed)**2)
        print(f'Lena reconstruction MSE: {lena_mse}')
    else:
        print('Lena image not found, skipping image test')
        
except ImportError as e:
    print(f'Missing dependencies: {e}')
except Exception as e:
    print(f'Error during image test: {e}')

print('Test completed')