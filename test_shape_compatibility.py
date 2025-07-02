#!/usr/bin/env python3
"""
Test suite for coefficient shape compatibility with MATLAB
Ensures all edge cases produce exactly matching array shapes
"""

import pytest
import numpy as np
import matlab.engine
from diffcurve.fdct2d import fdct_wrapping

class TestShapeCompatibility:
    """Test suite for exact shape compatibility with MATLAB"""
    
    @pytest.fixture(scope="class")
    def matlab_engine(self):
        """Shared MATLAB engine for all tests"""
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        yield eng
        eng.quit()
    
    def get_coefficient_shapes(self, img, matlab_engine):
        """Get coefficient shapes from both MATLAB and Python implementations"""
        
        # MATLAB
        img_matlab = matlab.double(img.tolist())
        coeffs_matlab = matlab_engine.fdct_wrapping(img_matlab, 0.0, 2.0, 3.0, 16.0)
        
        # Python
        coeffs_python = fdct_wrapping(img, is_real=0, finest=2, num_scales=3, num_angles_coarse=16)
        
        # Extract shapes
        matlab_shapes = []
        python_shapes = []
        
        for j in range(len(coeffs_matlab)):
            matlab_scale_shapes = []
            python_scale_shapes = []
            
            for l in range(len(coeffs_matlab[j])):
                matlab_shape = np.array(coeffs_matlab[j][l]).shape
                python_shape = coeffs_python[j][l].shape
                
                matlab_scale_shapes.append(matlab_shape)
                python_scale_shapes.append(python_shape)
            
            matlab_shapes.append(matlab_scale_shapes)
            python_shapes.append(python_scale_shapes)
        
        return matlab_shapes, python_shapes
    
    def test_original_failing_case_shapes(self, matlab_engine):
        """Test shapes for the original 63x66 failing case"""
        np.random.seed(100)
        img = np.random.randn(63, 66)
        
        matlab_shapes, python_shapes = self.get_coefficient_shapes(img, matlab_engine)
        
        # Verify all shapes match exactly
        mismatches = []
        for j in range(len(matlab_shapes)):
            for l in range(len(matlab_shapes[j])):
                if matlab_shapes[j][l] != python_shapes[j][l]:
                    mismatches.append((j, l, matlab_shapes[j][l], python_shapes[j][l]))
        
        assert len(mismatches) == 0, f"Shape mismatches in 63x66 case: {mismatches}"
        
        # Verify expected shapes for scale 1 (where issues occurred)
        expected_scale1_shapes = [
            (18, 24), (16, 22), (16, 22), (18, 24),  # Quadrant 1
            (22, 19), (22, 17), (22, 17), (22, 19),  # Quadrant 2
            (18, 24), (16, 22), (16, 22), (18, 24),  # Quadrant 3
            (22, 19), (22, 17), (22, 17), (22, 19),  # Quadrant 4
        ]
        
        actual_scale1_shapes = python_shapes[1]
        assert actual_scale1_shapes == expected_scale1_shapes, (
            f"Scale 1 shapes don't match expected pattern:\n"
            f"Expected: {expected_scale1_shapes}\n"
            f"Actual:   {actual_scale1_shapes}"
        )
    
    @pytest.mark.parametrize("height,width,seed,description", [
        (66, 69, 300, "Both dimensions divisible by 3"),
        (64, 66, 500, "Width divisible by 3"),
        (61, 67, 400, "Neither dimension divisible by 3"),
        (60, 63, 200, "Both divisible by 3, different ratio"),
        (72, 75, 600, "Large both divisible by 3"),
        (48, 51, 123, "Small both divisible by 3"),
        (45, 48, 456, "Medium both divisible by 3"),
    ])
    def test_various_dimension_shapes(self, matlab_engine, height, width, seed, description):
        """Test shapes for various dimension combinations"""
        np.random.seed(seed)
        img = np.random.randn(height, width)
        
        matlab_shapes, python_shapes = self.get_coefficient_shapes(img, matlab_engine)
        
        # Check all shapes match
        mismatches = []
        for j in range(len(matlab_shapes)):
            for l in range(len(matlab_shapes[j])):
                if matlab_shapes[j][l] != python_shapes[j][l]:
                    mismatches.append((j, l, matlab_shapes[j][l], python_shapes[j][l]))
        
        assert len(mismatches) == 0, (
            f"Shape mismatches in {description} ({height}x{width}):\n" +
            "\n".join([f"  Scale {j}, Angle {l}: MATLAB{m_shape} vs Python{p_shape}" 
                      for j, l, m_shape, p_shape in mismatches])
        )
    
    def test_quadrant_symmetry_patterns(self, matlab_engine):
        """Test that quadrant patterns are correct for edge cases"""
        test_cases = [
            (63, 66, 100),  # Original case
            (66, 69, 300),  # Another problematic case
        ]
        
        for height, width, seed in test_cases:
            np.random.seed(seed)
            img = np.random.randn(height, width)
            
            matlab_shapes, python_shapes = self.get_coefficient_shapes(img, matlab_engine)
            
            # Check scale 1 quadrant patterns (this is where edge cases manifest)
            scale1_shapes = python_shapes[1]
            
            # Each quadrant should have 4 angles
            assert len(scale1_shapes) == 16, f"Expected 16 angles at scale 1, got {len(scale1_shapes)}"
            
            # Extract quadrant shapes
            quad1_shapes = scale1_shapes[0:4]   # Angles 0-3
            quad2_shapes = scale1_shapes[4:8]   # Angles 4-7
            quad3_shapes = scale1_shapes[8:12]  # Angles 8-11
            quad4_shapes = scale1_shapes[12:16] # Angles 12-15
            
            # Quadrants 1 and 3 should have same pattern (odd quadrants)
            assert quad1_shapes == quad3_shapes, (
                f"Quadrants 1 and 3 should match for {height}x{width}:\n"
                f"Quad 1: {quad1_shapes}\n"
                f"Quad 3: {quad3_shapes}"
            )
            
            # Quadrants 2 and 4 should have same pattern (even quadrants)
            assert quad2_shapes == quad4_shapes, (
                f"Quadrants 2 and 4 should match for {height}x{width}:\n"
                f"Quad 2: {quad2_shapes}\n"
                f"Quad 4: {quad4_shapes}"
            )
            
            print(f"✅ {height}x{width}: Quadrant symmetry verified")
    
    def test_coefficient_count_consistency(self, matlab_engine):
        """Test that coefficient counts are consistent across implementations"""
        test_cases = [
            (63, 66, 100),
            (60, 60, 800),
            (72, 75, 600),
        ]
        
        for height, width, seed in test_cases:
            np.random.seed(seed)
            img = np.random.randn(height, width)
            
            matlab_shapes, python_shapes = self.get_coefficient_shapes(img, matlab_engine)
            
            # Verify same number of scales
            assert len(matlab_shapes) == len(python_shapes), (
                f"Scale count mismatch for {height}x{width}: "
                f"MATLAB={len(matlab_shapes)}, Python={len(python_shapes)}"
            )
            
            # Verify same number of angles per scale
            for j in range(len(matlab_shapes)):
                matlab_angle_count = len(matlab_shapes[j])
                python_angle_count = len(python_shapes[j])
                
                assert matlab_angle_count == python_angle_count, (
                    f"Angle count mismatch for {height}x{width} scale {j}: "
                    f"MATLAB={matlab_angle_count}, Python={python_angle_count}"
                )
            
            # Calculate total coefficient elements
            matlab_total_elements = sum(
                sum(np.prod(shape) for shape in scale_shapes)
                for scale_shapes in matlab_shapes
            )
            python_total_elements = sum(
                sum(np.prod(shape) for shape in scale_shapes)
                for scale_shapes in python_shapes
            )
            
            assert matlab_total_elements == python_total_elements, (
                f"Total element count mismatch for {height}x{width}: "
                f"MATLAB={matlab_total_elements}, Python={python_total_elements}"
            )

class TestShapeRegressionPrevention:
    """Prevent regression of shape compatibility fixes"""
    
    def test_no_shape_mismatches_in_critical_cases(self):
        """Ensure no shape mismatches in previously failing cases"""
        
        # These cases had shape mismatches before the fix
        critical_cases = [
            (63, 66, 100, "Original systematic failure case"),
            (66, 69, 300, "Both div by 3 with M=11.0,11.5"),
            (64, 66, 500, "Width div by 3 case"),
            (61, 67, 400, "Non-div-by-3 case that also failed"),
        ]
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            for height, width, seed, description in critical_cases:
                np.random.seed(seed)
                img = np.random.randn(height, width)
                
                # MATLAB
                img_matlab = matlab.double(img.tolist())
                coeffs_matlab = eng.fdct_wrapping(img_matlab, 0.0, 2.0, 3.0, 16.0)
                
                # Python
                coeffs_python = fdct_wrapping(img, is_real=0, finest=2, num_scales=3, num_angles_coarse=16)
                
                # Check every single coefficient shape
                total_coeffs = 0
                shape_mismatches = 0
                
                for j in range(len(coeffs_matlab)):
                    for l in range(len(coeffs_matlab[j])):
                        matlab_shape = np.array(coeffs_matlab[j][l]).shape
                        python_shape = coeffs_python[j][l].shape
                        
                        total_coeffs += 1
                        if matlab_shape != python_shape:
                            shape_mismatches += 1
                            print(f"SHAPE MISMATCH in {description}: Scale {j}, Angle {l}: MATLAB{matlab_shape} vs Python{python_shape}")
                
                assert shape_mismatches == 0, (
                    f"REGRESSION: {shape_mismatches}/{total_coeffs} shape mismatches in {description}"
                )
                
                print(f"✅ {description}: All {total_coeffs} coefficient shapes match")
        
        finally:
            eng.quit()

if __name__ == "__main__":
    print("Testing shape compatibility...")
    
    # Run shape compatibility tests
    test_shape = TestShapeCompatibility()
    eng = matlab.engine.start_matlab()
    eng.cd('diffcurve/fdct2d')
    
    try:
        test_shape.test_original_failing_case_shapes(eng)
        test_shape.test_quadrant_symmetry_patterns(eng)
        test_shape.test_coefficient_count_consistency(eng)
        print("✅ Shape compatibility tests passed")
        
    finally:
        eng.quit()
    
    # Run regression prevention tests
    test_regression = TestShapeRegressionPrevention()
    test_regression.test_no_shape_mismatches_in_critical_cases()
    print("✅ Shape regression prevention tests passed")
    
    print("🎉 All shape compatibility tests passed!")