#!/usr/bin/env python3
"""
Comprehensive test suite for curvelet transform implementation.

Tests the Python implementation against MATLAB ground truth results
to ensure numerical accuracy and correctness.
"""

import os
import numpy as np
import pytest
from pathlib import Path

from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping


class TestCurveletTransform:
    """Test suite for curvelet transform functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures and load reference data."""
        cls.test_data_dir = Path(__file__).parent / "reference_data"
        cls.tolerance = 1e-12  # Tolerance for numerical comparisons
        cls.matlab_tolerance = 1e-10  # Tolerance when comparing against MATLAB
        
    def test_perfect_reconstruction_64x64(self):
        """Test perfect reconstruction for 64x64 image."""
        np.random.seed(42)
        img = np.random.randn(64, 64)
        
        # Forward and inverse transform
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        # Check reconstruction error
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < self.tolerance, f"Perfect reconstruction failed: MSE = {mse}"
        
    def test_perfect_reconstruction_128x128(self):
        """Test perfect reconstruction for 128x128 image."""
        np.random.seed(123)
        img = np.random.randn(128, 128)
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < self.tolerance, f"Perfect reconstruction failed: MSE = {mse}"
        
    def test_perfect_reconstruction_256x256(self):
        """Test perfect reconstruction for 256x256 image."""
        np.random.seed(456)
        img = np.random.randn(256, 256)
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < self.tolerance, f"Perfect reconstruction failed: MSE = {mse}"
    
    def test_real_transform(self):
        """Test real-valued curvelet transform."""
        np.random.seed(789)
        img = np.random.randn(64, 64)
        
        # Real transform
        coeffs_real = fdct_wrapping(img, is_real=1)
        reconstructed_real = ifdct_wrapping(coeffs_real, is_real=1)
        
        # Ensure reconstruction is real
        assert np.all(np.isreal(reconstructed_real)), "Real transform should produce real output"
        
        # Check reconstruction error
        mse = np.mean(np.abs(img - reconstructed_real)**2)
        assert mse < self.tolerance, f"Real transform reconstruction failed: MSE = {mse}"
    
    def test_finest_parameter(self):
        """Test different finest parameter settings."""
        np.random.seed(111)
        img = np.random.randn(64, 64)
        
        # Test finest=1 (curvelets at finest level)
        coeffs1 = fdct_wrapping(img, finest=1)
        reconstructed1 = ifdct_wrapping(coeffs1)
        mse1 = np.mean(np.abs(img - reconstructed1)**2)
        
        # Test finest=2 (wavelets at finest level)
        coeffs2 = fdct_wrapping(img, finest=2)
        reconstructed2 = ifdct_wrapping(coeffs2)
        mse2 = np.mean(np.abs(img - reconstructed2)**2)
        
        assert mse1 < self.tolerance, f"Finest=1 reconstruction failed: MSE = {mse1}"
        assert mse2 < self.tolerance, f"Finest=2 reconstruction failed: MSE = {mse2}"
    
    def test_different_scales(self):
        """Test transforms with different number of scales."""
        np.random.seed(222)
        img = np.random.randn(128, 128)
        
        for num_scales in [3, 4, 5]:
            coeffs = fdct_wrapping(img, num_scales=num_scales)
            reconstructed = ifdct_wrapping(coeffs)
            mse = np.mean(np.abs(img - reconstructed)**2)
            
            assert mse < self.tolerance, f"Reconstruction failed for {num_scales} scales: MSE = {mse}"
            assert len(coeffs) == num_scales, f"Expected {num_scales} scales, got {len(coeffs)}"
    
    def test_different_angles(self):
        """Test transforms with different numbers of angles."""
        np.random.seed(333)
        img = np.random.randn(64, 64)
        
        for num_angles in [8, 16, 32]:
            coeffs = fdct_wrapping(img, num_angles_coarse=num_angles)
            reconstructed = ifdct_wrapping(coeffs)
            mse = np.mean(np.abs(img - reconstructed)**2)
            
            assert mse < self.tolerance, f"Reconstruction failed for {num_angles} angles: MSE = {mse}"
    
    def test_non_square_images(self):
        """Test transforms on non-square images."""
        np.random.seed(444)
        
        shapes = [(64, 128), (128, 64), (32, 96)]
        for height, width in shapes:
            img = np.random.randn(height, width)
            
            coeffs = fdct_wrapping(img)
            reconstructed = ifdct_wrapping(coeffs)
            mse = np.mean(np.abs(img - reconstructed)**2)
            
            assert mse < self.tolerance, f"Reconstruction failed for {height}x{width}: MSE = {mse}"
    
    def test_constant_image(self):
        """Test transform of constant image."""
        # Constant image should have energy concentrated in DC component
        img = np.ones((64, 64)) * 5.0
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < self.tolerance, f"Constant image reconstruction failed: MSE = {mse}"
    
    def test_delta_function(self):
        """Test transform of delta function."""
        # Delta function should be well-represented by curvelets
        img = np.zeros((64, 64))
        img[32, 32] = 1.0
        
        coeffs = fdct_wrapping(img)
        reconstructed = ifdct_wrapping(coeffs)
        
        mse = np.mean(np.abs(img - reconstructed)**2)
        assert mse < self.tolerance, f"Delta function reconstruction failed: MSE = {mse}"
    
    def test_coefficient_structure(self):
        """Test that coefficient structure is correct."""
        img = np.random.randn(64, 64)
        coeffs = fdct_wrapping(img, num_scales=4)
        
        # Check that we have the right number of scales
        assert len(coeffs) == 4, f"Expected 4 scales, got {len(coeffs)}"
        
        # Check that each scale has the right structure
        expected_angles = [1, 16, 32, 32]  # For default parameters
        for j, expected in enumerate(expected_angles):
            actual = len(coeffs[j])
            assert actual == expected, f"Scale {j}: expected {expected} angles, got {actual}"
    
    def test_linearity(self):
        """Test linearity of the transform."""
        np.random.seed(555)
        img1 = np.random.randn(64, 64)
        img2 = np.random.randn(64, 64)
        alpha, beta = 2.0, 3.0
        
        # Transform individual images
        coeffs1 = fdct_wrapping(img1)
        coeffs2 = fdct_wrapping(img2)
        
        # Transform linear combination
        img_combined = alpha * img1 + beta * img2
        coeffs_combined = fdct_wrapping(img_combined)
        
        # Check linearity: T(alpha*x + beta*y) = alpha*T(x) + beta*T(y)
        for j in range(len(coeffs1)):
            for l in range(len(coeffs1[j])):
                expected = alpha * coeffs1[j][l] + beta * coeffs2[j][l]
                actual = coeffs_combined[j][l]
                
                mse = np.mean(np.abs(expected - actual)**2)
                assert mse < self.tolerance, f"Linearity failed at scale {j}, angle {l}: MSE = {mse}"
    
    @pytest.mark.skipif(not os.path.exists("tests/reference_data"), 
                       reason="Reference data not available")
    def test_matlab_reference_random_64x64(self):
        """Test against MATLAB reference for random 64x64 image."""
        reference_file = self.test_data_dir / "matlab_random_64x64.npz"
        if not reference_file.exists():
            pytest.skip("MATLAB reference data not found")
            
        data = np.load(reference_file)
        img = data['image']
        matlab_reconstructed = data['reconstructed']
        
        # Python implementation
        coeffs = fdct_wrapping(img)
        python_reconstructed = ifdct_wrapping(coeffs)
        
        # Compare against MATLAB
        mse = np.mean(np.abs(matlab_reconstructed - python_reconstructed)**2)
        assert mse < self.matlab_tolerance, f"MATLAB comparison failed: MSE = {mse}"
    
    @pytest.mark.skipif(not os.path.exists("tests/reference_data"), 
                       reason="Reference data not available")
    def test_matlab_reference_various_sizes(self):
        """Test against MATLAB reference for various image sizes."""
        sizes = [32, 64, 128, 256]
        
        for size in sizes:
            reference_file = self.test_data_dir / f"matlab_random_{size}x{size}.npz"
            if not reference_file.exists():
                continue
                
            data = np.load(reference_file)
            img = data['image']
            matlab_reconstructed = data['reconstructed']
            
            coeffs = fdct_wrapping(img)
            python_reconstructed = ifdct_wrapping(coeffs)
            
            mse = np.mean(np.abs(matlab_reconstructed - python_reconstructed)**2)
            assert mse < self.matlab_tolerance, f"MATLAB comparison failed for {size}x{size}: MSE = {mse}"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])