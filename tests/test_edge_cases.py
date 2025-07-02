#!/usr/bin/env python3
"""
Edge case tests for curvelet transform implementation.
"""

import numpy as np
import pytest

from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimum_size_image(self):
        """Test smallest reasonable image size."""
        # 16x16 should be the practical minimum
        img = np.random.randn(16, 16)
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < 1e-10, f"Small image reconstruction failed: MSE = {mse}"
    
    def test_power_of_two_sizes(self):
        """Test various power-of-two image sizes."""
        sizes = [32, 64, 128, 256]
        
        for size in sizes:
            img = np.random.randn(size, size)
            
            coeffs = fdct_wrapping(img)
            reconstructed = ifdct_wrapping(coeffs)
            
            mse = np.mean(np.abs(img - reconstructed)**2)
            assert mse < 1e-12, f"Power-of-two size {size} failed: MSE = {mse}"
    
    def test_odd_sizes(self):
        """Test odd-sized images."""
        sizes = [33, 65, 127]
        
        for size in sizes:
            img = np.random.randn(size, size)
            
            coeffs = fdct_wrapping(img)
            reconstructed = ifdct_wrapping(coeffs)
            
            mse = np.mean(np.abs(img - reconstructed)**2)
            assert mse < 1e-12, f"Odd size {size} failed: MSE = {mse}"
    
    def test_extreme_values(self):
        """Test with extreme pixel values."""
        img = np.random.randn(64, 64)
        
        # Very large values
        img_large = img * 1e6
        coeffs = fdct_wrapping(img_large)
        reconstructed = ifdct_wrapping(coeffs)
        mse = np.mean(np.abs(img_large - reconstructed)**2) / np.mean(np.abs(img_large)**2)
        assert mse < 1e-12, f"Large values failed: relative MSE = {mse}"
        
        # Very small values
        img_small = img * 1e-6
        coeffs = fdct_wrapping(img_small)
        reconstructed = ifdct_wrapping(coeffs)
        mse = np.mean(np.abs(img_small - reconstructed)**2) / np.mean(np.abs(img_small)**2)
        assert mse < 1e-12, f"Small values failed: relative MSE = {mse}"
    
    def test_zero_image(self):
        """Test zero image."""
        img = np.zeros((64, 64))
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        assert np.allclose(reconstructed, 0), "Zero image should reconstruct to zero"
    
    def test_complex_input(self):
        """Test complex-valued input images."""
        img_real = np.random.randn(64, 64)
        img_imag = np.random.randn(64, 64)
        img_complex = img_real + 1j * img_imag
        
        coeffs = fdct_wrapping(img_complex)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img_complex - reconstructed)**2)
        assert mse < 1e-12, f"Complex input failed: MSE = {mse}"
    
    def test_invalid_parameters(self):
        """Test invalid parameter combinations."""
        img = np.random.randn(64, 64)
        
        # Invalid finest parameter
        with pytest.raises((ValueError, AssertionError)):
            fdct_wrapping(img, finest=3)
        
        # Invalid is_real parameter
        with pytest.raises((ValueError, AssertionError)):
            fdct_wrapping(img, is_real=2)
        
        # Too few scales
        with pytest.raises((ValueError, AssertionError)):
            fdct_wrapping(img, num_scales=1)
        
        # Non-multiple of 4 angles
        with pytest.raises((ValueError, AssertionError)):
            fdct_wrapping(img, num_angles_coarse=15)
    
    def test_memory_efficiency(self):
        """Test that large images don't cause memory issues."""
        # This test ensures our optimizations work for larger images
        img = np.random.randn(512, 512)
        
        try:
            coeffs = fdct_wrapping(img)
            reconstructed = ifdct_wrapping(coeffs)
            
            mse = np.mean(np.abs(img - reconstructed)**2)
            assert mse < 1e-12, f"Large image failed: MSE = {mse}"
        except MemoryError:
            pytest.skip("Not enough memory for 512x512 test")
    
    def test_numerical_stability(self):
        """Test numerical stability with challenging images."""
        # High-frequency checkerboard pattern
        x, y = np.meshgrid(np.arange(64), np.arange(64))
        checkerboard = ((x + y) % 2).astype(float)
        
        coeffs = fdct_wrapping(checkerboard)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(checkerboard - reconstructed)**2)
        assert mse < 1e-10, f"Checkerboard pattern failed: MSE = {mse}"
        
        # Random noise
        noise = np.random.randn(64, 64) * 0.1
        
        coeffs = fdct_wrapping(noise)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(noise - reconstructed)**2)
        assert mse < 1e-12, f"Random noise failed: MSE = {mse}"
    
    def test_coefficient_modification(self):
        """Test behavior when coefficients are modified."""
        img = np.random.randn(64, 64)
        coeffs = fdct_wrapping(img)
        
        # Zero out finest scale coefficients
        for l in range(len(coeffs[-1])):
            coeffs[-1][l] = np.zeros_like(coeffs[-1][l])
        
        # Should still reconstruct (but with loss of high frequencies)
        reconstructed = ifdct_wrapping(coeffs)
        assert reconstructed.shape == img.shape, "Shape should be preserved"
        assert np.all(np.isfinite(reconstructed)), "Reconstruction should be finite"
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        np.random.seed(12345)
        img1 = np.random.randn(64, 64)
        coeffs1 = fdct_wrapping(img1)
        
        np.random.seed(12345)
        img2 = np.random.randn(64, 64)
        coeffs2 = fdct_wrapping(img2)
        
        # Images should be identical
        assert np.allclose(img1, img2), "Random seeds should produce identical images"
        
        # Coefficients should be identical
        for j in range(len(coeffs1)):
            for l in range(len(coeffs1[j])):
                assert np.allclose(coeffs1[j][l], coeffs2[j][l]), f"Coefficients differ at scale {j}, angle {l}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])