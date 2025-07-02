#!/usr/bin/env python3
"""
Specific tests for wedge tick calculation precision
Tests the core algorithmic fix for MATLAB compatibility
"""

import pytest
import numpy as np
import matlab.engine
from diffcurve.fdct2d.fdct_wrapping import _compute_wedge_ticks, _matlab_round

class TestWedgeTickPrecision:
    """Test suite for wedge tick calculation precision"""
    
    def test_matlab_round_vs_python_round(self):
        """Test cases where MATLAB and Python rounding differ"""
        critical_cases = [
            # (value, matlab_result, python_np_round_result)
            (32.5, 33, 32),  # The critical failing case
            (6.5, 7, 6),     # Corner wedge calculation case
            (0.5, 1, 0),
            (1.5, 2, 2),     # This one actually matches
            (2.5, 3, 2),
            (10.5, 11, 10),
            (-0.5, 0, 0),    # Negative cases
            (-1.5, -1, -2),
            (-2.5, -2, -2),
        ]
        
        for value, matlab_expected, python_expected in critical_cases:
            matlab_result = _matlab_round(value)
            python_result = int(np.round(value))
            
            assert matlab_result == matlab_expected, f"_matlab_round({value}) = {matlab_result}, expected {matlab_expected}"
            assert python_result == python_expected, f"np.round({value}) = {python_result}, expected {python_expected}"
            
            # Show the difference for critical cases
            if matlab_expected != python_expected:
                print(f"Rounding difference: {value} -> MATLAB:{matlab_expected}, Python:{python_expected}")
    
    def test_wedge_tick_precision_problematic_m_values(self):
        """Test wedge tick calculation for M values that were problematic"""
        
        # These M values caused precision issues before the fix
        problematic_m_values = [
            10.5,   # 63/3/2 = 10.5 (from 63x66 case)
            11.0,   # 66/3/2 = 11.0 (from 63x66 case)
            5.25,   # M after scale division
            5.5,    # M after scale division
            10.166666666666666,  # From 61x67 case analysis
            11.166666666666666,  # From analysis
            11.5,   # From 66x69 case
        ]
        
        angles_per_quad = 4
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            for M_horiz in problematic_m_values:
                print(f"Testing M_horiz = {M_horiz}")
                
                # MATLAB reference calculation
                eng.eval(f"M_horiz = {M_horiz};", nargout=0)
                eng.eval("nbangles_perquad = 4;", nargout=0)
                
                # Execute MATLAB's exact algorithm
                if angles_per_quad % 2 == 0:  # Even case
                    eng.eval("wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);", nargout=0)
                    eng.eval("wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;", nargout=0)
                    eng.eval("wedge_ticks = [wedge_ticks_left, wedge_ticks_right((end-1):-1:1)];", nargout=0)
                else:  # Odd case
                    eng.eval("wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);", nargout=0)
                    eng.eval("wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;", nargout=0)
                    eng.eval("wedge_ticks = [wedge_ticks_left, wedge_ticks_right(end:-1:1)];", nargout=0)
                
                matlab_wedge_ticks = np.array(eng.eval("wedge_ticks")).flatten()
                
                # Python calculation with fix
                python_wedge_ticks = _compute_wedge_ticks(angles_per_quad, M_horiz)
                
                # Verify exact match
                assert np.array_equal(matlab_wedge_ticks, python_wedge_ticks), (
                    f"Wedge ticks mismatch for M_horiz={M_horiz}:\n"
                    f"MATLAB: {matlab_wedge_ticks}\n"
                    f"Python: {python_wedge_ticks}\n"
                    f"Diff:   {matlab_wedge_ticks - python_wedge_ticks}"
                )
                
                print(f"  ✅ Match: {matlab_wedge_ticks}")
        
        finally:
            eng.quit()
    
    def test_wedge_tick_before_and_after_rounding(self):
        """Test the exact values before and after rounding for critical cases"""
        
        # Focus on the case that was failing: M_horiz = 10.5
        M_horiz = 10.5
        angles_per_quad = 4
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            eng.eval(f"M_horiz = {M_horiz};", nargout=0)
            eng.eval("nbangles_perquad = 4;", nargout=0)
            
            # Get MATLAB's before-rounding values
            eng.eval("before_round = (0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1;", nargout=0)
            matlab_before_round = np.array(eng.eval("before_round")).flatten()
            
            # Get MATLAB's after-rounding values
            eng.eval("after_round = round(before_round);", nargout=0)
            matlab_after_round = np.array(eng.eval("after_round")).flatten()
            
        finally:
            eng.quit()
        
        # Python calculation
        floor_4M_horiz = int(np.floor(4*M_horiz))
        scale_factor = 2 * floor_4M_horiz
        step = 1.0 / (2 * angles_per_quad)
        sequence = np.arange(0, 0.5 + step, step)
        python_before_round = sequence * scale_factor + 1
        
        # Test both rounding methods
        python_np_round = np.round(python_before_round).astype(int)
        python_matlab_round = _matlab_round(python_before_round)
        
        print(f"M_horiz = {M_horiz}")
        print(f"Before rounding:")
        print(f"  MATLAB: {matlab_before_round}")
        print(f"  Python: {python_before_round}")
        print(f"After rounding:")
        print(f"  MATLAB: {matlab_after_round}")
        print(f"  Python np.round: {python_np_round}")
        print(f"  Python matlab_round: {python_matlab_round}")
        
        # Verify our before-rounding values match
        np.testing.assert_array_almost_equal(matlab_before_round, python_before_round, decimal=15)
        
        # Verify our MATLAB-style rounding matches MATLAB
        np.testing.assert_array_equal(matlab_after_round, python_matlab_round)
        
        # Show the difference with np.round
        if not np.array_equal(matlab_after_round, python_np_round):
            diff_indices = np.where(matlab_after_round != python_np_round)[0]
            print(f"Rounding differences at indices: {diff_indices}")
            for idx in diff_indices:
                print(f"  Index {idx}: value={python_before_round[idx]:.1f} -> MATLAB:{matlab_after_round[idx]}, np.round:{python_np_round[idx]}")

class TestCornerWedgeCalculations:
    """Test corner wedge calculations that also needed rounding fixes"""
    
    def test_first_wedge_endpoint_vert_calculation(self):
        """Test the first_wedge_endpoint_vert calculation that needed rounding fix"""
        
        # Test case from 63x66, scale 1, quadrant 2
        M_vert = 5.5
        angles_per_quad = 4
        
        eng = matlab.engine.start_matlab()
        eng.cd('diffcurve/fdct2d')
        
        try:
            eng.eval(f"M_vert = {M_vert};", nargout=0)
            eng.eval("nbangles_perquad = 4;", nargout=0)
            
            # MATLAB calculation
            eng.eval("first_wedge_endpoint_vert = round(2*floor(4*M_vert)/(2*nbangles_perquad) + 1);", nargout=0)
            matlab_result = eng.eval("first_wedge_endpoint_vert")
            
            # Get before-rounding value
            eng.eval("before_round = 2*floor(4*M_vert)/(2*nbangles_perquad) + 1;", nargout=0)
            matlab_before_round = eng.eval("before_round")
            
        finally:
            eng.quit()
        
        # Python calculation
        before_round = 2*int(np.floor(4*M_vert))/(2*angles_per_quad) + 1
        python_np_round = round(before_round)  # Standard Python round
        python_matlab_round = _matlab_round(before_round)
        
        print(f"first_wedge_endpoint_vert calculation for M_vert={M_vert}:")
        print(f"  Before round: MATLAB={matlab_before_round}, Python={before_round}")
        print(f"  After round: MATLAB={matlab_result}, Python round()={python_np_round}, _matlab_round()={python_matlab_round}")
        
        # Verify before-rounding values match
        assert abs(matlab_before_round - before_round) < 1e-15, f"Before-round mismatch: {matlab_before_round} vs {before_round}"
        
        # Verify our MATLAB-style rounding matches
        assert matlab_result == python_matlab_round, f"MATLAB round mismatch: {matlab_result} vs {python_matlab_round}"
        
        # Show if there would be a difference with standard Python rounding
        if matlab_result != python_np_round:
            print(f"  ⚠️ Standard Python round() would give wrong result: {python_np_round} instead of {matlab_result}")

if __name__ == "__main__":
    print("Testing wedge tick precision fixes...")
    
    test_wedge = TestWedgeTickPrecision()
    test_wedge.test_matlab_round_vs_python_round()
    test_wedge.test_wedge_tick_precision_problematic_m_values()
    test_wedge.test_wedge_tick_before_and_after_rounding()
    
    test_corner = TestCornerWedgeCalculations()
    test_corner.test_first_wedge_endpoint_vert_calculation()
    
    print("✅ All wedge tick precision tests passed!")