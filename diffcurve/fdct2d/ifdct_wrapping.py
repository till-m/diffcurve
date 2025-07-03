import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .fdct_wrapping_window import fdct_wrapping_window

# Cache commonly used constants
_SQRT_2 = np.sqrt(2)
_INV_SQRT_2 = 1.0 / _SQRT_2

# MATLAB-style rounding function (round half away from zero)
def _matlab_round(x):
    return np.floor(x + 0.5).astype(int)


def _bilinear_resize(array, new_shape):
    """
    Custom bilinear interpolation resize to replace scipy.ndimage.zoom.
    
    Parameters
    ----------
    array : ndarray
        Input array to resize
    new_shape : tuple
        Target shape (height, width)
        
    Returns
    -------
    ndarray
        Resized array using bilinear interpolation
    """
    old_height, old_width = array.shape
    new_height, new_width = new_shape
    
    # Create coordinate grids for the new shape
    y_new = np.linspace(0, old_height - 1, new_height)
    x_new = np.linspace(0, old_width - 1, new_width)
    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_new, y_new)
    
    # Get integer coordinates
    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    x1 = np.clip(x0 + 1, 0, old_width - 1)
    y1 = np.clip(y0 + 1, 0, old_height - 1)
    
    # Ensure coordinates are within bounds
    x0 = np.clip(x0, 0, old_width - 1)
    y0 = np.clip(y0, 0, old_height - 1)
    
    # Get fractional parts
    dx = x_grid - x0
    dy = y_grid - y0
    
    # Bilinear interpolation
    # f(x,y) ≈ f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy
    output = (array[y0, x0] * (1 - dx) * (1 - dy) +
              array[y0, x1] * dx * (1 - dy) +
              array[y1, x0] * (1 - dx) * dy +
              array[y1, x1] * dx * dy)
    
    return output


def ifdct_wrapping(coeffs, is_real=0, height=None, width=None):
    """
    Inverse Fast Discrete Curvelet Transform via wedge wrapping.
    
    Computes the adjoint/pseudo-inverse of the curvelet transform.
    Ported from ifdct_wrapping.m by Laurent Demanet, 2004.
    
    Parameters
    ----------
    coeffs : list of lists
        Curvelet coefficients from fdct_wrapping
    is_real : int, optional
        Transform type: 0 for complex, 1 for real (default: 0)
    height, width : int, optional
        Output image dimensions (not needed if finest=2)
    
    Returns
    -------
    x : ndarray
        Reconstructed image
    """
    # Initialization
    num_scales = len(coeffs)
    num_angles_coarse = len(coeffs[1])
    
    # Initialize angle counts for each scale
    num_angles = [1]
    scales_sequence = list(range(num_scales, 1, -1))
    for scale_val in scales_sequence:
        diff = num_scales - scale_val
        num_angles.append(num_angles_coarse * (2 ** int(np.ceil(diff / 2))))
    
    # Determine finest parameter
    if len(coeffs[-1]) == 1:
        finest = 2
    else:
        finest = 1
        
    if finest == 2:
        num_angles[num_scales - 1] = 1
    
    # Determine output dimensions
    if height is None or width is None:
        if finest == 1:
            raise ValueError('Height and width must be specified when finest=1')
        out_height, out_width = coeffs[-1][0].shape
    else:
        out_height = height
        out_width = width
    
    M1 = out_height / 3
    M2 = out_width / 3
    
    if finest == 1:
        bigN1 = 2 * int(np.floor(2 * M1)) + 1
        bigN2 = 2 * int(np.floor(2 * M2)) + 1
        X = np.zeros((bigN1, bigN2), dtype=complex)
        
        # Initialization: preparing the lowpass filter at finest scale
        window_length_1 = int(np.floor(2 * M1)) - int(np.floor(M1)) - 1 - (out_height % 3 == 0)
        window_length_2 = int(np.floor(2 * M2)) - int(np.floor(M2)) - 1 - (out_width % 3 == 0)
        coord_1 = np.linspace(0, 1, window_length_1 + 1)
        coord_2 = np.linspace(0, 1, window_length_2 + 1)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)
        lowpass_1 = np.concatenate([wl_1, np.ones(2 * int(np.floor(M1)) + 1), wr_1])
        if out_height % 3 == 0:
            lowpass_1 = np.concatenate([[0], lowpass_1, [0]])
        lowpass_2 = np.concatenate([wl_2, np.ones(2 * int(np.floor(M2)) + 1), wr_2])
        if out_width % 3 == 0:
            lowpass_2 = np.concatenate([[0], lowpass_2, [0]])
        lowpass = np.outer(lowpass_1, lowpass_2)
        
        scales = list(range(num_scales, 1, -1))
    else:
        M1 = M1 / 2
        M2 = M2 / 2
        
        bigN1 = 2 * int(np.floor(2 * M1)) + 1
        bigN2 = 2 * int(np.floor(2 * M2)) + 1
        X = np.zeros((bigN1, bigN2), dtype=complex)
        
        window_length_1 = int(np.floor(2 * M1)) - int(np.floor(M1)) - 1
        window_length_2 = int(np.floor(2 * M2)) - int(np.floor(M2)) - 1
        coord_1 = np.linspace(0, 1, window_length_1 + 1)
        coord_2 = np.linspace(0, 1, window_length_2 + 1)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)
        lowpass_1 = np.concatenate([wl_1, np.ones(2 * int(np.floor(M1)) + 1), wr_1])
        lowpass_2 = np.concatenate([wl_2, np.ones(2 * int(np.floor(M2)) + 1), wr_2])
        lowpass = np.outer(lowpass_1, lowpass_2)
        hipass_finest = np.sqrt(1 - lowpass**2)
        
        scales = list(range(num_scales - 1, 1, -1))
    
    # Loop: pyramidal reconstruction
    Xj_topleft_1 = 1  # MATLAB 1-based indexing
    Xj_topleft_2 = 1
    
    for j in scales:
        j_idx = j - 1  # Convert to 0-based indexing for Python
        
        M1 = M1 / 2
        M2 = M2 / 2
        window_length_1 = int(np.floor(2 * M1)) - int(np.floor(M1)) - 1
        window_length_2 = int(np.floor(2 * M2)) - int(np.floor(M2)) - 1
        coord_1 = np.linspace(0, 1, window_length_1 + 1)
        coord_2 = np.linspace(0, 1, window_length_2 + 1)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)
        lowpass_1 = np.concatenate([wl_1, np.ones(2 * int(np.floor(M1)) + 1), wr_1])
        lowpass_2 = np.concatenate([wl_2, np.ones(2 * int(np.floor(M2)) + 1), wr_2])
        lowpass_next = np.outer(lowpass_1, lowpass_2)
        hipass = np.sqrt(1 - lowpass_next**2)
        Xj = np.zeros((2 * int(np.floor(4 * M1)) + 1, 2 * int(np.floor(4 * M2)) + 1), dtype=complex)
        
        # Angular reconstruction
        angle_idx = 0
        num_quadrants = 2 + 2 * (not is_real)
        angles_per_quad = num_angles[j_idx] // 4
        
        for quadrant in range(1, num_quadrants + 1):
            M_horiz = M2 * (quadrant % 2 == 1) + M1 * (quadrant % 2 == 0)
            M_vert = M1 * (quadrant % 2 == 1) + M2 * (quadrant % 2 == 0)
            
            if angles_per_quad % 2:
                step = 0.5 / angles_per_quad
                wedge_ticks_left = _matlab_round(np.arange(0, 0.5 + step, step) * 2 * int(np.floor(4 * M_horiz)) + 1)
                wedge_ticks_right = 2 * int(np.floor(4 * M_horiz)) + 2 - wedge_ticks_left
                wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[::-1]])
            else:
                step = 0.5 / angles_per_quad
                wedge_ticks_left = _matlab_round(np.arange(0, 0.5 + step, step) * 2 * int(np.floor(4 * M_horiz)) + 1)
                wedge_ticks_right = 2 * int(np.floor(4 * M_horiz)) + 2 - wedge_ticks_left
                wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[-2::-1]])
            
            wedge_endpoints = wedge_ticks[1::2]  # 2:2:(end-1)
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2
            
            # Left corner wedge
            angle_idx += 1
            if angle_idx - 1 >= len(coeffs[j_idx]):
                continue
            first_wedge_endpoint_vert = _matlab_round(2 * int(np.floor(4 * M_vert)) / (2 * angles_per_quad) + 1)
            length_corner_wedge = int(np.floor(4 * M_vert)) - int(np.floor(M_vert)) + int(np.ceil(first_wedge_endpoint_vert / 4))
            Y_corner = np.arange(1, length_corner_wedge + 1)
            XX, YY = np.meshgrid(np.arange(1, 2 * int(np.floor(4 * M_horiz)) + 2), Y_corner)
            width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1
            slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[0]) / int(np.floor(4 * M_vert))
            left_line = _matlab_round(2 - wedge_endpoints[0] + slope_wedge * (Y_corner - 1))
            
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            first_row = int(np.floor(4 * M_vert)) + 2 - int(np.ceil((length_corner_wedge + 1) / 2)) + \
                       ((length_corner_wedge + 1) % 2) * (quadrant - 2 == (quadrant - 2) % 2)
            first_col = int(np.floor(4 * M_horiz)) + 2 - int(np.ceil((width_wedge + 1) / 2)) + \
                       ((width_wedge + 1) % 2) * (quadrant - 3 == (quadrant - 3) % 2)

            for row_idx, row in enumerate(Y_corner):
                width_range = np.arange(width_wedge)
                cols = left_line[row_idx] + np.mod(width_range - (left_line[row_idx] - first_col), width_wedge)
                new_row = 1 + (row - first_row) % length_corner_wedge
                admissible_cols = _matlab_round(0.5 * (cols + 1 + np.abs(cols - 1)))
                wrapped_XX[new_row - 1, :] = XX[row_idx, admissible_cols - 1]
                wrapped_YY[new_row - 1, :] = YY[row_idx, admissible_cols - 1]
            
            # Calculate window coordinates
            slope_wedge_right = (int(np.floor(4 * M_horiz)) + 1 - wedge_midpoints[0]) / int(np.floor(4 * M_vert))
            mid_line_right = wedge_midpoints[0] + slope_wedge_right * (wrapped_YY - 1)
            coord_right = 0.5 + int(np.floor(4 * M_vert)) / (wedge_endpoints[1] - wedge_endpoints[0]) * \
                         (wrapped_XX - mid_line_right) / (int(np.floor(4 * M_vert)) + 1 - wrapped_YY)
            
            C2 = 1 / (1 / (2 * int(np.floor(4 * M_horiz)) / (wedge_endpoints[0] - 1) - 1) + 
                     1 / (2 * int(np.floor(4 * M_vert)) / (first_wedge_endpoint_vert - 1) - 1))
            C1 = C2 / (2 * int(np.floor(4 * M_vert)) / (first_wedge_endpoint_vert - 1) - 1)
            
            mask = ((wrapped_XX - 1) / int(np.floor(4 * M_horiz)) + (wrapped_YY - 1) / int(np.floor(4 * M_vert))) == 2
            wrapped_XX[mask] = wrapped_XX[mask] + 1
            
            coord_corner = C1 + C2 * ((wrapped_XX - 1) / int(np.floor(4 * M_horiz)) - (wrapped_YY - 1) / int(np.floor(4 * M_vert))) / \
                          (2 - ((wrapped_XX - 1) / int(np.floor(4 * M_horiz)) + (wrapped_YY - 1) / int(np.floor(4 * M_vert))))
            
            wl_left = fdct_wrapping_window(coord_corner)[0]
            wr_right = fdct_wrapping_window(coord_right)[1]
            
            # Get coefficient and transform
            if is_real == 0:
                coeff = coeffs[j_idx][angle_idx - 1]
                wrapped_data = fftshift(fft2(ifftshift(coeff))) / np.sqrt(coeff.size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                if angle_idx - 1 + num_angles[j_idx] // 2 < len(coeffs[j_idx]):
                    x = coeffs[j_idx][angle_idx - 1] + 1j * coeffs[j_idx][angle_idx - 1 + num_angles[j_idx] // 2]
                    wrapped_data = fftshift(fft2(ifftshift(x))) * _INV_SQRT_2 / np.sqrt(x.size)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    continue
            
            # Ensure window functions match wrapped_data shape
            if wl_left.shape != wrapped_data.shape:
                target_shape = wrapped_data.shape
                wl_left = _bilinear_resize(wl_left, target_shape)
                wr_right = _bilinear_resize(wr_right, target_shape)
            
            wrapped_data = wrapped_data * (wl_left * wr_right)
            
            # Unwrapping data
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = _matlab_round(0.5 * (cols + 1 + np.abs(cols - 1)))
                new_row = 1 + (row - first_row) % length_corner_wedge
                
                # Handle shape mismatch between admissible_cols and wrapped_data
                if new_row - 1 < wrapped_data.shape[0]:
                    n_cols = min(len(admissible_cols), wrapped_data.shape[1])
                    valid_indices = (admissible_cols[:n_cols] >= 1) & (admissible_cols[:n_cols] <= Xj.shape[1])
                    if np.any(valid_indices):
                        Xj[row - 1, admissible_cols[:n_cols][valid_indices] - 1] += wrapped_data[new_row - 1, :n_cols][valid_indices]
            
            # Regular wedges
            length_wedge = int(np.floor(4 * M_vert)) - int(np.floor(M_vert))
            Y = np.arange(1, length_wedge + 1)
            first_row = int(np.floor(4 * M_vert)) + 2 - int(np.ceil((length_wedge + 1) / 2)) + \
                       ((length_wedge + 1) % 2) * (quadrant - 2 == (quadrant - 2) % 2)
            
            for subl in range(2, angles_per_quad):
                angle_idx += 1
                if angle_idx - 1 >= len(coeffs[j_idx]):
                    continue
                width_wedge = wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1
                slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[subl - 1]) / int(np.floor(4 * M_vert))
                left_line = _matlab_round(wedge_endpoints[subl - 2] + slope_wedge * (Y - 1))
                
                wrapped_XX = np.zeros((length_wedge, width_wedge))
                wrapped_YY = np.zeros((length_wedge, width_wedge))
                first_col = int(np.floor(4 * M_horiz)) + 2 - int(np.ceil((width_wedge + 1) / 2)) + \
                           ((width_wedge + 1) % 2) * (quadrant - 3 == (quadrant - 3) % 2)
                
                for row_idx, row in enumerate(Y):
                    cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                    new_row = 1 + (row - first_row) % length_wedge
                    wrapped_XX[new_row - 1, :] = XX[row_idx, cols - 1]
                    wrapped_YY[new_row - 1, :] = YY[row_idx, cols - 1]
                
                # Window calculations
                slope_wedge_left = (int(np.floor(4 * M_horiz)) + 1 - wedge_midpoints[subl - 2]) / int(np.floor(4 * M_vert))
                mid_line_left = wedge_midpoints[subl - 2] + slope_wedge_left * (wrapped_YY - 1)
                coord_left = 0.5 + int(np.floor(4 * M_vert)) / (wedge_endpoints[subl - 1] - wedge_endpoints[subl - 2]) * \
                            (wrapped_XX - mid_line_left) / (int(np.floor(4 * M_vert)) + 1 - wrapped_YY)
                slope_wedge_right = (int(np.floor(4 * M_horiz)) + 1 - wedge_midpoints[subl - 1]) / int(np.floor(4 * M_vert))
                mid_line_right = wedge_midpoints[subl - 1] + slope_wedge_right * (wrapped_YY - 1)
                coord_right = 0.5 + int(np.floor(4 * M_vert)) / (wedge_endpoints[subl] - wedge_endpoints[subl - 1]) * \
                             (wrapped_XX - mid_line_right) / (int(np.floor(4 * M_vert)) + 1 - wrapped_YY)
                
                wl_left = fdct_wrapping_window(coord_left)[0]
                wr_right = fdct_wrapping_window(coord_right)[1]
                
                # Get coefficient and transform
                if is_real == 0:
                    wrapped_data = fftshift(fft2(ifftshift(coeffs[j_idx][angle_idx - 1]))) / np.sqrt(coeffs[j_idx][angle_idx - 1].size)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    x = coeffs[j_idx][angle_idx - 1] + 1j * coeffs[j_idx][angle_idx - 1 + num_angles[j_idx] // 2]
                    wrapped_data = fftshift(fft2(ifftshift(x))) / np.sqrt(x.size) / np.sqrt(2)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                
                # Ensure window functions match wrapped_data shape
                if wl_left.shape != wrapped_data.shape:
                    from scipy.ndimage import zoom
                    zoom_factor_0 = wrapped_data.shape[0] / wl_left.shape[0]
                    zoom_factor_1 = wrapped_data.shape[1] / wl_left.shape[1]
                    wl_left = zoom(wl_left, (zoom_factor_0, zoom_factor_1), order=1)
                    wr_right = zoom(wr_right, (zoom_factor_0, zoom_factor_1), order=1)
                
                wrapped_data = wrapped_data * (wl_left * wr_right)
                
                # Unwrapping data
                for row_idx, row in enumerate(Y):
                    cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                    new_row = 1 + (row - first_row) % length_wedge
                    
                    # Handle shape mismatch between cols and wrapped_data
                    if new_row - 1 < wrapped_data.shape[0]:
                        n_cols = min(len(cols), wrapped_data.shape[1])
                        valid_indices = (cols[:n_cols] >= 1) & (cols[:n_cols] <= Xj.shape[1])
                        if np.any(valid_indices):
                            Xj[row - 1, cols[:n_cols][valid_indices] - 1] += wrapped_data[new_row - 1, :n_cols][valid_indices]
            
            # Right corner wedge
            angle_idx += 1
            if angle_idx - 1 >= len(coeffs[j_idx]):
                continue
            width_wedge = 4 * int(np.floor(4 * M_horiz)) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2]
            slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[-1]) / int(np.floor(4 * M_vert))
            left_line = _matlab_round(wedge_endpoints[-2] + slope_wedge * (Y_corner - 1))
            
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            first_row = int(np.floor(4 * M_vert)) + 2 - int(np.ceil((length_corner_wedge + 1) / 2)) + \
                       ((length_corner_wedge + 1) % 2) * (quadrant - 2 == (quadrant - 2) % 2)
            first_col = int(np.floor(4 * M_horiz)) + 2 - int(np.ceil((width_wedge + 1) / 2)) + \
                       ((width_wedge + 1) % 2) * (quadrant - 3 == (quadrant - 3) % 2)
            
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = _matlab_round(0.5 * (cols + 2 * int(np.floor(4 * M_horiz)) + 1 - 
                                                 np.abs(cols - (2 * int(np.floor(4 * M_horiz)) + 1))))
                new_row = 1 + (row - first_row) % length_corner_wedge
                wrapped_XX[new_row - 1, :] = XX[row_idx, admissible_cols - 1]
                wrapped_YY[new_row - 1, :] = YY[row_idx, admissible_cols - 1]
            
            # Coordinate calculations - MATLAB line 252: YY = Y_corner'*ones(1,width_wedge);
            YY_corner = np.outer(Y_corner, np.ones(width_wedge))
            slope_wedge_left = (int(np.floor(4 * M_horiz)) + 1 - wedge_midpoints[-1]) / int(np.floor(4 * M_vert))
            mid_line_left = wedge_midpoints[-1] + slope_wedge_left * (wrapped_YY - 1)
            coord_left = 0.5 + int(np.floor(4 * M_vert)) / (wedge_endpoints[-1] - wedge_endpoints[-2]) * \
                        (wrapped_XX - mid_line_left) / (int(np.floor(4 * M_vert)) + 1 - wrapped_YY)
            
            C2 = -1 / (2 * int(np.floor(4 * M_horiz)) / (wedge_endpoints[-1] - 1) - 1 + 
                      1 / (2 * int(np.floor(4 * M_vert)) / (first_wedge_endpoint_vert - 1) - 1))
            C1 = -C2 * (2 * int(np.floor(4 * M_horiz)) / (wedge_endpoints[-1] - 1) - 1)
            
            mask = ((wrapped_XX - 1) / int(np.floor(4 * M_horiz))) == ((wrapped_YY - 1) / int(np.floor(4 * M_vert)))
            wrapped_XX[mask] = wrapped_XX[mask] - 1
            
            coord_corner = C1 + C2 * (2 - ((wrapped_XX - 1) / int(np.floor(4 * M_horiz)) + (wrapped_YY - 1) / int(np.floor(4 * M_vert)))) / \
                          ((wrapped_XX - 1) / int(np.floor(4 * M_horiz)) - (wrapped_YY - 1) / int(np.floor(4 * M_vert)))
            
            wl_left = fdct_wrapping_window(coord_left)[0]
            wr_right = fdct_wrapping_window(coord_corner)[1]
            
            # Get coefficient and transform
            if is_real == 0:
                coeff = coeffs[j_idx][angle_idx - 1]
                wrapped_data = fftshift(fft2(ifftshift(coeff))) / np.sqrt(coeff.size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                if angle_idx - 1 + num_angles[j_idx] // 2 < len(coeffs[j_idx]):
                    x = coeffs[j_idx][angle_idx - 1] + 1j * coeffs[j_idx][angle_idx - 1 + num_angles[j_idx] // 2]
                    wrapped_data = fftshift(fft2(ifftshift(x))) * _INV_SQRT_2 / np.sqrt(x.size)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    continue
            
            # Ensure window functions match wrapped_data shape
            if wl_left.shape != wrapped_data.shape:
                target_shape = wrapped_data.shape
                wl_left = _bilinear_resize(wl_left, target_shape)
                wr_right = _bilinear_resize(wr_right, target_shape)
            
            wrapped_data = wrapped_data * (wl_left * wr_right)
            
            # Unwrapping data - MATLAB line 281: Xj(row,fliplr(admissible_cols)) = Xj(row,fliplr(admissible_cols)) + wrapped_data(new_row,end:-1:1);
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = _matlab_round(0.5 * (cols + 2 * int(np.floor(4 * M_horiz)) + 1 - 
                                                 np.abs(cols - (2 * int(np.floor(4 * M_horiz)) + 1))))
                new_row = 1 + (row - first_row) % length_corner_wedge
                flipped_cols = np.flip(admissible_cols)
                
                # Handle shape mismatch between flipped_cols and wrapped_data
                if new_row - 1 < wrapped_data.shape[0]:
                    n_cols = min(len(flipped_cols), wrapped_data.shape[1])
                    valid_indices = (flipped_cols[:n_cols] >= 1) & (flipped_cols[:n_cols] <= Xj.shape[1])
                    if np.any(valid_indices):
                        Xj[row - 1, flipped_cols[:n_cols][valid_indices] - 1] += wrapped_data[new_row - 1, ::-1][:n_cols][valid_indices]
            
            Xj = np.rot90(Xj)
        
        # Apply filters and accumulate
        Xj = Xj * lowpass
        # Apply hipass filter to central region - center on current center to preserve DC
        current_center_1 = Xj.shape[0] // 2
        current_center_2 = Xj.shape[1] // 2
        half_size_1 = int(np.floor(2 * M1))
        half_size_2 = int(np.floor(2 * M2))
        
        Xj_index1 = np.arange(current_center_1 - half_size_1, current_center_1 + half_size_1 + 1)
        Xj_index2 = np.arange(current_center_2 - half_size_2, current_center_2 + half_size_2 + 1)
        Xj[np.ix_(Xj_index1, Xj_index2)] = Xj[np.ix_(Xj_index1, Xj_index2)] * hipass
        
        # MATLAB: loc_1 = Xj_topleft_1 + (0:(2*floor(4*M1)));
        loc_1 = Xj_topleft_1 + np.arange(2 * int(np.floor(4 * M1)) + 1) - 1  # Convert to 0-based
        loc_2 = Xj_topleft_2 + np.arange(2 * int(np.floor(4 * M2)) + 1) - 1
        X[np.ix_(loc_1, loc_2)] = X[np.ix_(loc_1, loc_2)] + Xj
        
        # Preparing for loop reentry or exit
        Xj_topleft_1 = Xj_topleft_1 + int(np.floor(4 * M1)) - int(np.floor(2 * M1))
        Xj_topleft_2 = Xj_topleft_2 + int(np.floor(4 * M2)) - int(np.floor(2 * M2))
        
        lowpass = lowpass_next
    
    if is_real:
        Y = X.copy()
        X = np.rot90(X, 2)
        X = X + np.conj(Y)
    
    # Coarsest wavelet level
    M1 = M1 / 2
    M2 = M2 / 2
    Xj = fftshift(fft2(ifftshift(coeffs[0][0]))) / np.sqrt(coeffs[0][0].size)
    loc_1 = Xj_topleft_1 + np.arange(2 * int(np.floor(4 * M1)) + 1) - 1
    loc_2 = Xj_topleft_2 + np.arange(2 * int(np.floor(4 * M2)) + 1) - 1
    X[np.ix_(loc_1, loc_2)] = X[np.ix_(loc_1, loc_2)] + Xj * lowpass
    
    # Finest level
    M1 = out_height / 3
    M2 = out_width / 3
    if finest == 1:
        # Folding back onto output matrix
        shift_1 = int(np.floor(2 * M1)) - int(np.floor(out_height / 2))
        shift_2 = int(np.floor(2 * M2)) - int(np.floor(out_width / 2))
        
        Y = X[:, shift_2:shift_2 + out_width]
        Y[:, out_width - shift_2:out_width] = Y[:, out_width - shift_2:out_width] + X[:, :shift_2]
        Y[:, :shift_2] = Y[:, :shift_2] + X[:, out_width + shift_2:out_width + 2 * shift_2]
        X = Y[shift_1:shift_1 + out_height, :]
        X[out_height - shift_1:out_height, :] = X[out_height - shift_1:out_height, :] + Y[:shift_1, :]
        X[:shift_1, :] = X[:shift_1, :] + Y[out_height + shift_1:out_height + 2 * shift_1, :]
    else:
        # Extension to output matrix
        Y = fftshift(fft2(ifftshift(coeffs[num_scales - 1][0]))) / np.sqrt(coeffs[num_scales - 1][0].size)
        # Fix indexing for 0-based Python vs 1-based MATLAB
        X_topleft_1 = out_height // 2 - int(np.floor(M1))
        X_topleft_2 = out_width // 2 - int(np.floor(M2))
        loc_1 = X_topleft_1 + np.arange(2 * int(np.floor(M1)) + 1)
        loc_2 = X_topleft_2 + np.arange(2 * int(np.floor(M2)) + 1)
        Y[np.ix_(loc_1, loc_2)] = Y[np.ix_(loc_1, loc_2)] * hipass_finest + X
        X = Y
    
    x = fftshift(ifft2(ifftshift(X))) * np.sqrt(X.size)
    if is_real:
        x = np.real(x)
    
    return x