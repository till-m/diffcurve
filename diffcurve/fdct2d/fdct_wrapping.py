import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from math import ceil, log2
from .fdct_wrapping_window import fdct_wrapping_window

# Cache commonly used constants
_SQRT_2 = np.sqrt(2)
_HALF = 0.5

# MATLAB-style rounding function (round half away from zero)
def _matlab_round(x):
    return np.floor(x + 0.5).astype(int)


def _create_lowpass_filter(M1, M2, height=None, width=None):
    """Create lowpass filter for curvelet transform."""
    # Cache floor operations
    floor_2M1 = int(np.floor(2*M1))
    floor_M1 = int(np.floor(M1))
    floor_2M2 = int(np.floor(2*M2))
    floor_M2 = int(np.floor(M2))
    
    window_length_1 = floor_2M1 - floor_M1 - 1
    window_length_2 = floor_2M2 - floor_M2 - 1
    
    if height is not None and height % 3 == 0:
        window_length_1 -= 1
    if width is not None and width % 3 == 0:
        window_length_2 -= 1
    
    coord_1 = np.linspace(0, 1, window_length_1 + 1)
    coord_2 = np.linspace(0, 1, window_length_2 + 1)
    
    wl_1, wr_1 = fdct_wrapping_window(coord_1)
    wl_2, wr_2 = fdct_wrapping_window(coord_2)
    
    # Use cached values and pre-allocate concatenation
    ones_1 = np.ones(2*floor_M1 + 1)
    ones_2 = np.ones(2*floor_M2 + 1)
    
    lowpass_1 = np.concatenate([wl_1, ones_1, wr_1])
    lowpass_2 = np.concatenate([wl_2, ones_2, wr_2])
    
    if height is not None and height % 3 == 0:
        lowpass_1 = np.concatenate([[0], lowpass_1, [0]])
    if width is not None and width % 3 == 0:
        lowpass_2 = np.concatenate([[0], lowpass_2, [0]])
    
    return np.outer(lowpass_1, lowpass_2)


def _compute_wedge_ticks(angles_per_quad, M_horiz):
    """
    Compute wedge tick positions for angular decomposition.
    
    Exactly replicates MATLAB's algorithm:
    wedge_ticks_left = round((0:(1/(2*nbangles_perquad)):.5)*2*floor(4*M_horiz) + 1);
    wedge_ticks_right = 2*floor(4*M_horiz) + 2 - wedge_ticks_left;
    """
    # Cache expensive floor operation
    floor_4M_horiz = int(np.floor(4*M_horiz))
    scale_factor = 2 * floor_4M_horiz
    
    # MATLAB: (0:(1/(2*nbangles_perquad)):.5)
    # This creates a sequence from 0 to 0.5 with step 1/(2*nbangles_perquad)
    step = 1.0 / (2 * angles_per_quad)
    sequence = np.arange(0, 0.5 + step, step)
    
    # MATLAB: round(...*2*floor(4*M_horiz) + 1)
    # Use MATLAB-style rounding (round half away from zero) instead of Python's round half to even
    wedge_ticks_left = _matlab_round(sequence * scale_factor + 1)
    
    # MATLAB: 2*floor(4*M_horiz) + 2 - wedge_ticks_left
    wedge_ticks_right = scale_factor + 2 - wedge_ticks_left
    
    # MATLAB conditional for odd/even angles_per_quad
    if angles_per_quad % 2:
        # MATLAB: [wedge_ticks_left, wedge_ticks_right(end:-1:1)]
        wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[::-1]])
    else:
        # MATLAB: [wedge_ticks_left, wedge_ticks_right((end-1):-1:1)]
        wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[-2::-1]])
        
    return wedge_ticks


def fdct_wrapping(x, is_real=0, finest=2, num_scales=None, num_angles_coarse=16):
    """
    Fast Discrete Curvelet Transform via wedge wrapping.
    
    Ported from fdct_wrapping.m by Laurent Demanet, 2004.
    
    Parameters
    ----------
    x : ndarray
        Input matrix (M x N)
    is_real : int, optional
        Transform type: 0 for complex curvelets, 1 for real curvelets (default: 0)
    finest : int, optional
        Finest level coefficients: 1 for curvelets, 2 for wavelets (default: 2)
    num_scales : int, optional
        Number of scales including coarsest level. Default: ceil(log2(min(M,N)) - 3)
    num_angles_coarse : int, optional
        Number of angles at 2nd coarsest level, must be multiple of 4 (default: 16)
    
    Returns
    -------
    coeffs : list of lists
        Curvelet coefficients. coeffs[j][l] contains coefficients at scale j, angle l.
        For real transforms, cosine coefficients are in first two quadrants,
        sine coefficients in last two quadrants.
    """
    # Parameter validation
    if finest not in [1, 2]:
        raise ValueError(f"finest must be 1 or 2, got {finest}")
    
    if is_real not in [0, 1]:
        raise ValueError(f"is_real must be 0 or 1, got {is_real}")
    
    if num_angles_coarse % 4 != 0:
        raise ValueError(f"num_angles_coarse must be multiple of 4, got {num_angles_coarse}")
    
    # Take the 2D FFT
    X = fftshift(fft2(ifftshift(x))) / np.sqrt(x.size)
    height, width = X.shape
    
    # Set default parameters
    if num_scales is None:
        num_scales = int(ceil(log2(min(height, width)) - 3))
    
    if num_scales < 2:
        raise ValueError(f"num_scales must be at least 2, got {num_scales}")
    
    # Initialize angle counts for each scale
    num_angles = [1]
    scales_sequence = list(range(num_scales, 1, -1))
    for scale_val in scales_sequence:
        diff = num_scales - scale_val
        num_angles.append(num_angles_coarse * (2 ** int(ceil(diff / 2))))
    
    if finest == 2:
        num_angles[num_scales - 1] = 1
    
    # Initialize coefficient array
    coeffs = []
    for j in range(num_scales):
        scale_coeffs = [None] * num_angles[j]
        coeffs.append(scale_coeffs)
    
    # Pyramidal scale decomposition
    M1 = height / 3
    M2 = width / 3
    
    if finest == 1:
        # Initialization: smooth periodic extension of high frequencies
        bigN1 = 2 * int(np.floor(2 * M1)) + 1
        bigN2 = 2 * int(np.floor(2 * M2)) + 1
        
        equiv_index_1 = np.mod(int(np.floor(height/2)) - int(np.floor(2*M1)) + np.arange(bigN1), height)
        equiv_index_2 = np.mod(int(np.floor(width/2)) - int(np.floor(2*M2)) + np.arange(bigN2), width)
        
        X = X[np.ix_(equiv_index_1, equiv_index_2)]
        
        window_length_1 = int(np.floor(2*M1)) - int(np.floor(M1)) - 1 - (height % 3 == 0)
        window_length_2 = int(np.floor(2*M2)) - int(np.floor(M2)) - 1 - (width % 3 == 0)
        
        coord_1 = np.linspace(0, 1, window_length_1 + 1)
        coord_2 = np.linspace(0, 1, window_length_2 + 1)
        
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)
        
        lowpass_1 = np.concatenate([wl_1, np.ones(2*int(np.floor(M1))+1), wr_1])
        if height % 3 == 0:
            lowpass_1 = np.concatenate([[0], lowpass_1, [0]])
        
        lowpass_2 = np.concatenate([wl_2, np.ones(2*int(np.floor(M2))+1), wr_2])
        if width % 3 == 0:
            lowpass_2 = np.concatenate([[0], lowpass_2, [0]])
        
        lowpass = np.outer(lowpass_1, lowpass_2)
        Xlow = X * lowpass
        
        scales = list(range(num_scales, 1, -1))
    
    else:
        M1 = M1 / 2
        M2 = M2 / 2
        lowpass = _create_lowpass_filter(M1, M2)
        hipass = np.sqrt(1 - lowpass**2)
        
        Xlow_index_1 = np.arange(-int(np.floor(2*M1)), int(np.floor(2*M1))+1) + int(np.ceil((height+1)/2)) - 1
        Xlow_index_2 = np.arange(-int(np.floor(2*M2)), int(np.floor(2*M2))+1) + int(np.ceil((width+1)/2)) - 1
        
        Xlow = X[np.ix_(Xlow_index_1, Xlow_index_2)] * lowpass
        Xhi = X.copy()
        Xhi[np.ix_(Xlow_index_1, Xlow_index_2)] = Xhi[np.ix_(Xlow_index_1, Xlow_index_2)] * hipass
        
        coeffs[num_scales-1][0] = fftshift(ifft2(ifftshift(Xhi))) * np.sqrt(Xhi.size)
        if is_real:
            coeffs[num_scales-1][0] = np.real(coeffs[num_scales-1][0])
        
        scales = list(range(num_scales - 1, 1, -1))
    
    # Main loop over scales
    for j in scales:
        j_idx = j - 1  # Convert to 0-based indexing
        
        M1 = M1 / 2
        M2 = M2 / 2
        lowpass = _create_lowpass_filter(M1, M2)
        hipass = np.sqrt(1 - lowpass**2)
        
        Xhi = Xlow.copy()  # size is 2*floor(4*M1)+1 - by - 2*floor(4*M2)+1
        # Extract central region - center on the current center to preserve DC component
        current_center_1 = Xlow.shape[0] // 2
        current_center_2 = Xlow.shape[1] // 2
        half_size_1 = int(np.floor(2*M1))
        half_size_2 = int(np.floor(2*M2))
        
        Xlow_index_1 = np.arange(current_center_1 - half_size_1, current_center_1 + half_size_1 + 1)
        Xlow_index_2 = np.arange(current_center_2 - half_size_2, current_center_2 + half_size_2 + 1)
        
        Xlow = Xlow[np.ix_(Xlow_index_1, Xlow_index_2)]
        Xhi[np.ix_(Xlow_index_1, Xlow_index_2)] = Xlow * hipass
        Xlow = Xlow * lowpass  # size is 2*floor(2*M1)+1 - by - 2*floor(2*M2)+1
        
        # Angular decomposition
        angle_idx = 0
        num_quadrants = 2 + 2 * (not is_real)
        angles_per_quad = num_angles[j_idx] // 4
        
        for quadrant in range(1, num_quadrants + 1):
            # Use bit operations for faster parity check
            is_odd_quadrant = quadrant & 1
            M_horiz = M2 * is_odd_quadrant + M1 * (1 - is_odd_quadrant)
            M_vert = M1 * is_odd_quadrant + M2 * (1 - is_odd_quadrant)
            
            # Cache expensive floor operations for this quadrant
            floor_4M_vert = int(np.floor(4*M_vert))
            floor_4M_horiz = int(np.floor(4*M_horiz))
            floor_M_vert = int(np.floor(M_vert))
            
            wedge_ticks = _compute_wedge_ticks(angles_per_quad, M_horiz)
                
            wedge_endpoints = wedge_ticks[1:-1:2]  # 2:2:(end-1)
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2
            
            # Left corner wedge
            angle_idx += 1
            first_wedge_endpoint_vert = _matlab_round(2*int(np.floor(4*M_vert))/(2*angles_per_quad) + 1)
            length_corner_wedge = int(np.floor(4*M_vert)) - int(np.floor(M_vert)) + int(np.ceil(first_wedge_endpoint_vert/4))
            Y_corner = np.arange(1, length_corner_wedge + 1)
            XX, YY = np.meshgrid(np.arange(1, 2*int(np.floor(4*M_horiz)) + 2), Y_corner)
            
            width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1
            slope_wedge = (int(np.floor(4*M_horiz)) + 1 - wedge_endpoints[0]) / int(np.floor(4*M_vert))
            left_line = np.floor(2 - wedge_endpoints[0] + slope_wedge * (Y_corner - 1) + 0.5).astype(int)
            
            wrapped_data = np.zeros((length_corner_wedge, width_wedge), dtype=complex)
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            
            first_row = floor_4M_vert + 2 - int(np.ceil((length_corner_wedge+1)/2)) + \
                       ((length_corner_wedge+1) % 2) * (quadrant-2 == (quadrant-2) % 2)
            first_col = floor_4M_horiz + 2 - int(np.ceil((width_wedge+1)/2)) + \
                       ((width_wedge+1) % 2) * (quadrant-3 == (quadrant-3) % 2)
            
            for row_idx, row in enumerate(Y_corner):
                width_range = np.arange(width_wedge)
                cols = left_line[row_idx] + np.mod(width_range - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = np.floor(0.5 * (cols + 1 + np.abs(cols - 1)) + 0.5).astype(int)
                new_row = 1 + (row - first_row) % length_corner_wedge
                
                cols_mask = cols > 0
                wrapped_data[new_row-1, :] = Xhi[row-1, admissible_cols-1] * cols_mask
                wrapped_XX[new_row-1, :] = XX[row_idx, admissible_cols-1]
                wrapped_YY[new_row-1, :] = YY[row_idx, admissible_cols-1]
            
            # Apply windowing - use cached values
            slope_wedge_right = (floor_4M_horiz + 1 - wedge_midpoints[0]) / floor_4M_vert
            mid_line_right = wedge_midpoints[0] + slope_wedge_right * (wrapped_YY - 1)
            
            coord_right = _HALF + floor_4M_vert / (wedge_endpoints[1] - wedge_endpoints[0]) * \
                         (wrapped_XX - mid_line_right) / (floor_4M_vert + 1 - wrapped_YY)
            
            # Pre-compute frequently used terms
            inv_floor_4M_horiz = 1.0 / floor_4M_horiz
            inv_floor_4M_vert = 1.0 / floor_4M_vert
            
            C2 = 1 / (1 / (2*floor_4M_horiz/(wedge_endpoints[0] - 1) - 1) + 
                     1 / (2*floor_4M_vert/(first_wedge_endpoint_vert - 1) - 1))
            C1 = C2 / (2*floor_4M_vert/(first_wedge_endpoint_vert - 1) - 1)
            
            # Cache coordinate calculations
            wrapped_XX_norm = (wrapped_XX - 1) * inv_floor_4M_horiz
            wrapped_YY_norm = (wrapped_YY - 1) * inv_floor_4M_vert
            
            mask = (wrapped_XX_norm + wrapped_YY_norm) == 2
            wrapped_XX[mask] = wrapped_XX[mask] + 1
            
            # Recalculate normalized coordinates after mask application
            wrapped_XX_norm = (wrapped_XX - 1) * inv_floor_4M_horiz
            coord_corner = C1 + C2 * (wrapped_XX_norm - wrapped_YY_norm) / \
                          (2 - (wrapped_XX_norm + wrapped_YY_norm))
            
            wl_left = fdct_wrapping_window(coord_corner)[0]
            wr_right = fdct_wrapping_window(coord_right)[1]
            
            # Apply windowing and rotation in one step to avoid extra array operations
            wrapped_data *= (wl_left * wr_right)
            wrapped_data = np.rot90(wrapped_data, -(quadrant-1))
            
            # Optimize FFT operations - cache size calculation
            sqrt_size = np.sqrt(wrapped_data.size)
            
            if is_real == 0:
                coeffs[j_idx][angle_idx-1] = fftshift(ifft2(ifftshift(wrapped_data))) * sqrt_size
            else:
                x_temp = fftshift(ifft2(ifftshift(wrapped_data))) * sqrt_size
                coeffs[j_idx][angle_idx-1] = _SQRT_2 * np.real(x_temp)
                coeffs[j_idx][angle_idx-1+num_angles[j_idx]//2] = _SQRT_2 * np.imag(x_temp)
            
            # Regular wedges
            length_wedge = int(np.floor(4*M_vert)) - int(np.floor(M_vert))
            Y = np.arange(1, length_wedge + 1)
            first_row = int(np.floor(4*M_vert)) + 2 - int(np.ceil((length_wedge+1)/2)) + \
                       ((length_wedge+1) % 2) * (quadrant-2 == (quadrant-2) % 2)
            
            for subl in range(2, angles_per_quad):
                angle_idx += 1
                width_wedge = wedge_endpoints[subl] - wedge_endpoints[subl-2] + 1
                slope_wedge = (floor_4M_horiz + 1 - wedge_endpoints[subl-1]) / floor_4M_vert
                left_line = np.floor(wedge_endpoints[subl-2] + slope_wedge * (Y - 1) + 0.5).astype(int)
                
                wrapped_data = np.zeros((length_wedge, width_wedge), dtype=complex)
                wrapped_XX = np.zeros((length_wedge, width_wedge))
                wrapped_YY = np.zeros((length_wedge, width_wedge))
                
                first_col = floor_4M_horiz + 2 - int(np.ceil((width_wedge+1)/2)) + \
                           ((width_wedge+1) % 2) * (quadrant-3 == (quadrant-3) % 2)
                
                for row_idx, row in enumerate(Y):
                    width_range = np.arange(width_wedge)
                    cols = left_line[row_idx] + np.mod(width_range - (left_line[row_idx] - first_col), width_wedge)
                    new_row = 1 + (row - first_row) % length_wedge

                    wrapped_data[new_row-1, :] = Xhi[row-1, cols-1]
                    wrapped_XX[new_row-1, :] = XX[row_idx, cols-1]
                    wrapped_YY[new_row-1, :] = YY[row_idx, cols-1]
                
                # Apply windowing for regular wedge - use cached values
                slope_wedge_left = (floor_4M_horiz + 1 - wedge_midpoints[subl-2]) / floor_4M_vert
                mid_line_left = wedge_midpoints[subl-2] + slope_wedge_left * (wrapped_YY - 1)
                coord_left = _HALF + floor_4M_vert / (wedge_endpoints[subl-1] - wedge_endpoints[subl-2]) * \
                            (wrapped_XX - mid_line_left) / (floor_4M_vert + 1 - wrapped_YY)
                
                slope_wedge_right = (floor_4M_horiz + 1 - wedge_midpoints[subl-1]) / floor_4M_vert
                mid_line_right = wedge_midpoints[subl-1] + slope_wedge_right * (wrapped_YY - 1)
                coord_right = _HALF + floor_4M_vert / (wedge_endpoints[subl] - wedge_endpoints[subl-1]) * \
                             (wrapped_XX - mid_line_right) / (floor_4M_vert + 1 - wrapped_YY)
                
                wl_left = fdct_wrapping_window(coord_left)[0]
                wr_right = fdct_wrapping_window(coord_right)[1]
                
                wrapped_data = wrapped_data * (wl_left * wr_right)
                
                if is_real == 0:
                    wrapped_data = np.rot90(wrapped_data, -(quadrant-1))
                    coeffs[j_idx][angle_idx-1] = fftshift(ifft2(ifftshift(wrapped_data))) * np.sqrt(wrapped_data.size)
                else:
                    wrapped_data = np.rot90(wrapped_data, -(quadrant-1))
                    x_temp = fftshift(ifft2(ifftshift(wrapped_data))) * np.sqrt(wrapped_data.size)
                    coeffs[j_idx][angle_idx-1] = np.sqrt(2) * np.real(x_temp)
                    coeffs[j_idx][angle_idx-1+num_angles[j_idx]//2] = np.sqrt(2) * np.imag(x_temp)
            
            # Right corner wedge
            angle_idx += 1
            width_wedge = 4*int(np.floor(4*M_horiz)) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2]
            slope_wedge = (int(np.floor(4*M_horiz)) + 1 - wedge_endpoints[-1]) / int(np.floor(4*M_vert))
            left_line = np.floor(wedge_endpoints[-2] + slope_wedge * (Y_corner - 1) + 0.5).astype(int)
            
            wrapped_data = np.zeros((length_corner_wedge, width_wedge), dtype=complex)
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            
            first_row = floor_4M_vert + 2 - int(np.ceil((length_corner_wedge+1)/2)) + \
                       ((length_corner_wedge+1) % 2) * (quadrant-2 == (quadrant-2) % 2)
            first_col = floor_4M_horiz + 2 - int(np.ceil((width_wedge+1)/2)) + \
                       ((width_wedge+1) % 2) * (quadrant-3 == (quadrant-3) % 2)
            

            max_col = 2*floor_4M_horiz + 1
            
            for row_idx, row in enumerate(Y_corner):
                width_range = np.arange(width_wedge)
                cols = left_line[row_idx] + np.mod(width_range - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = np.floor(0.5 * (cols + max_col - np.abs(cols - max_col)) + 0.5).astype(int)
                new_row = 1 + (row - first_row) % length_corner_wedge

                cols_mask = cols <= max_col
                wrapped_data[new_row-1, :] = Xhi[row-1, admissible_cols-1] * cols_mask
                wrapped_XX[new_row-1, :] = XX[row_idx, admissible_cols-1]
                wrapped_YY[new_row-1, :] = YY[row_idx, admissible_cols-1]
            
            # Apply windowing for right corner wedge - use cached values
            slope_wedge_left = (floor_4M_horiz + 1 - wedge_midpoints[-1]) / floor_4M_vert
            mid_line_left = wedge_midpoints[-1] + slope_wedge_left * (wrapped_YY - 1)
            coord_left = _HALF + floor_4M_vert / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
                        (wrapped_XX - mid_line_left) / (floor_4M_vert + 1 - wrapped_YY)
            
            C2 = -1 / (2*floor_4M_horiz/(wedge_endpoints[-1] - 1) - 1 + 
                      1 / (2*floor_4M_vert/(first_wedge_endpoint_vert - 1) - 1))
            C1 = -C2 * (2*floor_4M_horiz/(wedge_endpoints[-1] - 1) - 1)
            
            # Use pre-computed inverse values for efficiency
            wrapped_XX_norm = (wrapped_XX - 1) * inv_floor_4M_horiz
            wrapped_YY_norm = (wrapped_YY - 1) * inv_floor_4M_vert
            
            mask = wrapped_XX_norm == wrapped_YY_norm
            wrapped_XX[mask] = wrapped_XX[mask] - 1
            
            # Recalculate after mask modification
            wrapped_XX_norm = (wrapped_XX - 1) * inv_floor_4M_horiz
            coord_corner = C1 + C2 * (2 - (wrapped_XX_norm + wrapped_YY_norm)) / \
                          (wrapped_XX_norm - wrapped_YY_norm)
            
            wl_left = fdct_wrapping_window(coord_left)[0]
            wr_right = fdct_wrapping_window(coord_corner)[1]
            
            wrapped_data = wrapped_data * (wl_left * wr_right)
            wrapped_data = np.rot90(wrapped_data, -(quadrant-1))
            
            if is_real == 0:
                coeffs[j_idx][angle_idx-1] = fftshift(ifft2(ifftshift(wrapped_data))) * np.sqrt(wrapped_data.size)
            else:
                x_temp = fftshift(ifft2(ifftshift(wrapped_data))) * np.sqrt(wrapped_data.size)
                coeffs[j_idx][angle_idx-1] = np.sqrt(2) * np.real(x_temp)
                coeffs[j_idx][angle_idx-1+num_angles[j_idx]//2] = np.sqrt(2) * np.imag(x_temp)
            
            if quadrant < num_quadrants:
                Xhi = np.rot90(Xhi)
    
    # Coarsest wavelet level
    coeffs[0][0] = fftshift(ifft2(ifftshift(Xlow))) * np.sqrt(Xlow.size)
    if is_real == 1:
        coeffs[0][0] = np.real(coeffs[0][0])
    
    return coeffs