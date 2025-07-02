#!/usr/bin/env python3
"""
Comprehensive machine precision tests for curvelet transform
Tests all edge cases that were fixed for MATLAB compatibility
"""

import pytest
import numpy as np
import matlab.engine
from diffcurve.fdct2d import fdct_wrapping

# Machine precision tolerance
MACHINE_PRECISION_TOLERANCE = 3e-15

class TestMachinePrecision:
    """Test suite for machine precision compatibility with MATLAB"""
    
    @pytest.fixture(scope="class")
    def matlab_engine(self):
        """Shared MATLAB engine for all tests"""
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        yield eng
        eng.quit()
    
    def compare_with_matlab(self, img, matlab_engine, description=""):
        """Compare Python implementation with MATLAB reference"""
        
        # MATLAB computation
        img_matlab = matlab.double(img.tolist())
        coeffs_matlab = matlab_engine.fdct_wrapping(img_matlab, 0.0, 2.0, 3.0, 16.0)
        
        # Python computation
        coeffs_python = fdct_wrapping(img, is_real=0, finest=2, num_scales=3, num_angles_coarse=16)
        
        # Validate shapes first
        shape_mismatches = []
        for j in range(len(coeffs_matlab)):
            for l in range(len(coeffs_matlab[j])):
                matlab_shape = np.array(coeffs_matlab[j][l]).shape
                python_shape = coeffs_python[j][l].shape
                if matlab_shape != python_shape:
                    shape_mismatches.append((j, l, matlab_shape, python_shape))
        
        assert len(shape_mismatches) == 0, f"Shape mismatches in {description}: {shape_mismatches}"
        
        # Validate precision
        max_error = 0
        precision_failures = []
        
        for j in range(len(coeffs_matlab)):
            for l in range(len(coeffs_matlab[j])):
                matlab_coeff = np.array(coeffs_matlab[j][l])
                python_coeff = coeffs_python[j][l]
                
                abs_diff = np.abs(matlab_coeff - python_coeff)
                max_abs_diff = np.max(abs_diff)
                max_error = max(max_error, max_abs_diff)
                
                if max_abs_diff > MACHINE_PRECISION_TOLERANCE:
                    precision_failures.append((j, l, max_abs_diff))
        
        assert len(precision_failures) == 0, (
            f"Precision failures in {description} (tolerance={MACHINE_PRECISION_TOLERANCE}): "
            f"max_error={max_error:.2e}, failures={precision_failures}"
        )
        
        return max_error
    
    def test_original_problematic_case(self, matlab_engine):
        """Test the original 63x66 case that had systematic shape mismatches"""
        np.random.seed(100)
        img = np.random.randn(63, 66)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "63x66 original problematic case")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 63x66 case: max_error = {max_error:.2e}")
    
    def test_both_dimensions_divisible_by_3_case1(self, matlab_engine):
        """Test 66x69 case (both dimensions divisible by 3, M values 11.0, 11.5)"""
        np.random.seed(300)
        img = np.random.randn(66, 69)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "66x69 both div by 3")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 66x69 case: max_error = {max_error:.2e}")
    
    def test_width_divisible_by_3(self, matlab_engine):
        """Test 64x66 case (width divisible by 3)"""
        np.random.seed(500)
        img = np.random.randn(64, 66)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "64x66 width div by 3")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 64x66 case: max_error = {max_error:.2e}")
    
    def test_neither_dimension_divisible_by_3(self, matlab_engine):
        """Test 61x67 case (neither dimension divisible by 3)"""
        np.random.seed(400)
        img = np.random.randn(61, 67)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "61x67 neither div by 3")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 61x67 case: max_error = {max_error:.2e}")
    
    def test_square_both_divisible_by_3(self, matlab_engine):
        """Test 60x60 case (square, both dimensions divisible by 3)"""
        np.random.seed(800)
        img = np.random.randn(60, 60)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "60x60 square div by 3")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 60x60 case: max_error = {max_error:.2e}")
    
    def test_large_both_divisible_by_3(self, matlab_engine):
        """Test 72x75 case (larger dimensions, both divisible by 3)"""
        np.random.seed(600)
        img = np.random.randn(72, 75)
        
        max_error = self.compare_with_matlab(img, matlab_engine, "72x75 large div by 3")
        assert max_error <= MACHINE_PRECISION_TOLERANCE
        print(f"✅ 72x75 case: max_error = {max_error:.2e}")

class TestRoundingFix:
    """Test suite specifically for the rounding fixes"""
    
    def test_matlab_rounding_function(self):
        """Test that our MATLAB-style rounding matches MATLAB exactly"""
        from diffcurve.fdct2d.fdct_wrapping import _matlab_round
        
        # Test cases where Python and MATLAB rounding differ
        test_cases = [
            (32.5, 33),  # The critical case that was failing
            (6.5, 7),    # Another case from corner wedge calculation
            (0.5, 1),
            (1.5, 2),
            (-0.5, 0),   # Negative cases
            (-1.5, -1),
            (10.0, 10),  # Exact integers
            (10.1, 10),  # Non-half cases
            (10.9, 11),
        ]
        
        for input_val, expected in test_cases:
            result = _matlab_round(input_val)
            assert result == expected, f"_matlab_round({input_val}) = {result}, expected {expected}"
    
    def test_wedge_tick_calculation_precision(self):
        """Test that wedge tick calculations now match MATLAB exactly"""
        from diffcurve.fdct2d.fdct_wrapping import _compute_wedge_ticks
        
        # Test the specific M_horiz values that were problematic
        problematic_m_horiz_values = [10.5, 11.0, 5.25, 5.5]
        angles_per_quad = 4
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            for M_horiz in problematic_m_horiz_values:
                # MATLAB calculation
                eng.eval(f"M_horiz = {M_horiz};", nargout=0)
                eng.eval("nbangles_perquad = 4;", nargout=0)
                eng.eval("wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);", nargout=0)
                eng.eval("wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;", nargout=0)
                eng.eval("wedge_ticks = [wedge_ticks_left, wedge_ticks_right((end-1):-1:1)];", nargout=0)
                
                matlab_wedge_ticks = np.array(eng.eval("wedge_ticks")).flatten()
                
                # Python calculation
                python_wedge_ticks = _compute_wedge_ticks(angles_per_quad, M_horiz)
                
                assert np.array_equal(matlab_wedge_ticks, python_wedge_ticks), (
                    f"Wedge ticks mismatch for M_horiz={M_horiz}: "
                    f"MATLAB={matlab_wedge_ticks}, Python={python_wedge_ticks}"
                )
        
        finally:
            eng.quit()

class TestEdgeCaseRegression:
    """Regression tests to ensure edge cases remain fixed"""
    
    @pytest.mark.parametrize("height,width,seed,description", [
        (63, 66, 100, "Original failing case"),
        (66, 69, 300, "Both div by 3 case"),
        (64, 66, 500, "Width div by 3 case"),
        (61, 67, 400, "Neither div by 3 case"),
        (48, 51, 123, "Small both div by 3"),
        (75, 78, 789, "Large both div by 3"),
        (45, 48, 456, "Medium both div by 3"),
    ])
    def test_edge_case_regression(self, height, width, seed, description):
        """Regression test for various edge cases"""
        np.random.seed(seed)
        img = np.random.randn(height, width)
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            # MATLAB
            img_matlab = matlab.double(img.tolist())
            coeffs_matlab = eng.fdct_wrapping(img_matlab, 0.0, 2.0, 3.0, 16.0)
            
            # Python
            coeffs_python = fdct_wrapping(img, is_real=0, finest=2, num_scales=3, num_angles_coarse=16)
            
            # Check all shapes match
            for j in range(len(coeffs_matlab)):
                for l in range(len(coeffs_matlab[j])):
                    matlab_shape = np.array(coeffs_matlab[j][l]).shape
                    python_shape = coeffs_python[j][l].shape
                    assert matlab_shape == python_shape, (
                        f"Shape mismatch in {description} at scale {j}, angle {l}: "
                        f"MATLAB{matlab_shape} vs Python{python_shape}"
                    )
            
            # Check precision
            max_error = 0
            for j in range(len(coeffs_matlab)):
                for l in range(len(coeffs_matlab[j])):
                    matlab_coeff = np.array(coeffs_matlab[j][l])
                    python_coeff = coeffs_python[j][l]
                    
                    abs_diff = np.abs(matlab_coeff - python_coeff)
                    max_abs_diff = np.max(abs_diff)
                    max_error = max(max_error, max_abs_diff)
            
            assert max_error <= MACHINE_PRECISION_TOLERANCE, (
                f"Precision failure in {description}: max_error={max_error:.2e} > {MACHINE_PRECISION_TOLERANCE}"
            )
        
        finally:
            eng.quit()

if __name__ == "__main__":
    # Run tests directly
    import sys
    
    print("Running machine precision tests...")
    
    # Test rounding function
    test_rounding = TestRoundingFix()
    test_rounding.test_matlab_rounding_function()
    test_rounding.test_wedge_tick_calculation_precision()
    print("✅ Rounding fix tests passed")
    
    # Test machine precision cases
    test_precision = TestMachinePrecision()
    eng = matlab.engine.start_matlab()
    eng.cd('diffcurve/fdct2d')
    
    try:
        test_precision.test_original_problematic_case(eng)
        test_precision.test_both_dimensions_divisible_by_3_case1(eng)
        test_precision.test_width_divisible_by_3(eng)
        test_precision.test_neither_dimension_divisible_by_3(eng)
        test_precision.test_square_both_divisible_by_3(eng)
        test_precision.test_large_both_divisible_by_3(eng)
        print("✅ All machine precision tests passed")
        
    finally:
        eng.quit()
    
    print("🎉 ALL TESTS PASSED - MACHINE PRECISION ACHIEVED")