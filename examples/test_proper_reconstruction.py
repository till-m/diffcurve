import cv2
import numpy as np
from pathlib import Path
from diffcurve.utils import get_project_root
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping

def test_proper_reconstruction():
    """Test proper forward → inverse reconstruction"""
    
    project_root = get_project_root()
    lena_file = Path.joinpath(project_root, "data/Lena.jpg")
    
    if lena_file.exists():
        lena_img = cv2.imread(str(lena_file), 0).astype(float) / 255
        print(f'Lena image shape: {lena_img.shape}')
        
        dct_kwargs = {
            'is_real': 0,
            'finest': 2,
            'nbscales': 6,
            'nbangles_coarse': 16
        }
        
        print("Testing proper forward → inverse reconstruction...")
        
        # Forward transform
        coeffs = fdct_wrapping(lena_img, **dct_kwargs)
        
        # Inverse transform
        reconstructed = ifdct_wrapping(coeffs, is_real=dct_kwargs['is_real'])
        
        # Calculate MSE
        mse = np.mean(np.abs(lena_img - reconstructed)**2)
        print(f'Forward → Inverse MSE: {mse}')
        
        if mse < 1e-10:
            print('✅ Excellent reconstruction quality')
        elif mse < 1e-6:
            print('✅ Good reconstruction quality')  
        elif mse < 1e-3:
            print('⚠️  Acceptable reconstruction quality')
        else:
            print('❌ Poor reconstruction quality')
            
        return mse
    else:
        print('Lena image not found')
        return None

def test_individual_curvelet_reconstruction():
    """Test the unusual individual curvelet reconstruction approach from the original test"""
    
    project_root = get_project_root()
    lena_file = Path.joinpath(project_root, "data/Lena.jpg")
    
    if lena_file.exists():
        lena_img = cv2.imread(str(lena_file), 0).astype(float) / 255
        
        dct_kwargs = {
            'is_real': 0,
            'finest': 2,
            'nbscales': 6,
            'nbangles_coarse': 16
        }
        
        print("\nTesting individual curvelet reconstruction (original test approach)...")
        
        # Get coefficients for the image
        img_coeffs = fdct_wrapping(lena_img, **dct_kwargs)
        
        # Create zero coefficients template
        zeros = np.zeros_like(lena_img)
        zero_coeffs = fdct_wrapping(zeros, **dct_kwargs)
        
        # Reconstruct individual curvelets and sum them
        all_reconstructions = []
        total_coeffs = 0
        
        for (scale_idx, curvelets_scale) in enumerate(zero_coeffs):
            for (wedge_idx, curvelet_wedge) in enumerate(curvelets_scale):
                total_coeffs += 1
                
                # Create temporary coefficient array with only this one coefficient
                tmp = [
                    [np.zeros_like(coeff) for coeff in scale]
                    for scale in zero_coeffs
                ]
                
                # Set only this one coefficient
                tmp[scale_idx][wedge_idx] = img_coeffs[scale_idx][wedge_idx].copy()
                
                # Reconstruct this individual curvelet
                reconstruction = ifdct_wrapping(tmp, is_real=dct_kwargs['is_real'])
                all_reconstructions.append(reconstruction)
        
        print(f'Total coefficients processed: {total_coeffs}')
        
        # Sum all individual reconstructions
        summed_reconstruction = np.sum(all_reconstructions, axis=0)
        
        # Calculate MSE
        mse = np.mean(np.abs(lena_img - summed_reconstruction.real)**2)
        print(f'Individual curvelet sum MSE: {mse}')
        
        if mse < 1e-10:
            print('✅ Excellent reconstruction quality')
        elif mse < 1e-6:
            print('✅ Good reconstruction quality')  
        elif mse < 1e-3:
            print('⚠️  Acceptable reconstruction quality')
        else:
            print('❌ Poor reconstruction quality')
            
        return mse
    else:
        print('Lena image not found')
        return None

if __name__ == "__main__":
    proper_mse = test_proper_reconstruction()
    individual_mse = test_individual_curvelet_reconstruction()
    
    print(f"\n=== Summary ===")
    if proper_mse is not None:
        print(f"Proper reconstruction MSE: {proper_mse}")
    if individual_mse is not None:
        print(f"Individual curvelet sum MSE: {individual_mse}")
        
    if proper_mse is not None and individual_mse is not None:
        if proper_mse < 1e-10 and individual_mse > 1e-3:
            print("✅ The algorithm works correctly!")
            print("❌ The test approach is flawed - individual curvelet reconstruction doesn't work well")
        elif proper_mse > 1e-3:
            print("❌ The algorithm has issues with basic reconstruction")
        else:
            print("Both approaches work well")