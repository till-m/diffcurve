import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from .fdct_wrapping_window import fdct_wrapping_window


def ifdct_wrapping(C, is_real=0, M=None, N=None):
    """
    Inverse Fast Discrete Curvelet Transform via wedge wrapping - Version 1.0
    This is in fact the adjoint, also the pseudo-inverse
    
    This is a direct Python port of ifdct_wrapping.m by Laurent Demanet, 2004.
    
    Parameters:
    -----------
    C : list of lists
        Cell array containing curvelet coefficients (see description in fdct_wrapping)
    is_real : int, optional
        As used in fdct_wrapping (default: 0)
    M, N : int, optional
        Size of the image to be recovered (not necessary if finest = 2)
    
    Returns:
    --------
    x : ndarray
        M-by-N matrix
    """
    # Initialization
    nbscales = len(C)
    nbangles_coarse = len(C[1])
    
    # MATLAB: nbangles = [1, nbangles_coarse .* 2.^(ceil((nbscales-(nbscales:-1:2))/2))];
    # nbscales:-1:2 means [nbscales, nbscales-1, ..., 2]
    # So nbscales-(nbscales:-1:2) means [0, 1, 2, ..., nbscales-2]
    nbangles = [1]
    scales_sequence = list(range(nbscales, 1, -1))  # nbscales:-1:2
    for k, scale_val in enumerate(scales_sequence):
        diff = nbscales - scale_val  # This gives [0, 1, 2, ..., nbscales-2]
        nbangles.append(nbangles_coarse * (2 ** int(np.ceil(diff / 2))))
    
    # Determine finest parameter
    if len(C[-1]) == 1:
        finest = 2
    else:
        finest = 1
        
    if finest == 2:
        nbangles[nbscales - 1] = 1
    
    # Determine output dimensions
    if M is None or N is None:
        if finest == 1:
            raise ValueError('Syntax: ifdct_wrapping(C, is_real, M, N) where the matrix to be recovered is M-by-N')
        N1, N2 = C[-1][0].shape
    else:
        N1 = M
        N2 = N
    
    M1 = N1 / 3
    M2 = N2 / 3
    
    if finest == 1:
        bigN1 = 2 * int(np.floor(2 * M1)) + 1
        bigN2 = 2 * int(np.floor(2 * M2)) + 1
        X = np.zeros((bigN1, bigN2), dtype=complex)
        
        # Initialization: preparing the lowpass filter at finest scale
        window_length_1 = int(np.floor(2 * M1)) - int(np.floor(M1)) - 1 - (N1 % 3 == 0)
        window_length_2 = int(np.floor(2 * M2)) - int(np.floor(M2)) - 1 - (N2 % 3 == 0)
        coord_1 = np.linspace(0, 1, window_length_1 + 1)
        coord_2 = np.linspace(0, 1, window_length_2 + 1)
        wl_1, wr_1 = fdct_wrapping_window(coord_1)
        wl_2, wr_2 = fdct_wrapping_window(coord_2)
        lowpass_1 = np.concatenate([wl_1, np.ones(2 * int(np.floor(M1)) + 1), wr_1])
        if N1 % 3 == 0:
            lowpass_1 = np.concatenate([[0], lowpass_1, [0]])
        lowpass_2 = np.concatenate([wl_2, np.ones(2 * int(np.floor(M2)) + 1), wr_2])
        if N2 % 3 == 0:
            lowpass_2 = np.concatenate([[0], lowpass_2, [0]])
        lowpass = np.outer(lowpass_1, lowpass_2)
        
        scales = list(range(nbscales, 1, -1))  # nbscales:-1:2
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
        
        scales = list(range(nbscales - 1, 1, -1))  # (nbscales-1):-1:2
    
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
        
        # Loop: angles
        l = 0
        nbquadrants = 2 + 2 * (not is_real)
        nbangles_perquad = nbangles[j_idx] // 4
        
        for quadrant in range(1, nbquadrants + 1):
            M_horiz = M2 * (quadrant % 2 == 1) + M1 * (quadrant % 2 == 0)
            M_vert = M1 * (quadrant % 2 == 1) + M2 * (quadrant % 2 == 0)
            
            if nbangles_perquad % 2:
                step = 0.5 / nbangles_perquad
                wedge_ticks_left = np.round(np.arange(0, 0.5 + step, step) * 2 * int(np.floor(4 * M_horiz)) + 1).astype(int)
                wedge_ticks_right = 2 * int(np.floor(4 * M_horiz)) + 2 - wedge_ticks_left
                wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[::-1]])
            else:
                step = 0.5 / nbangles_perquad
                wedge_ticks_left = np.round(np.arange(0, 0.5 + step, step) * 2 * int(np.floor(4 * M_horiz)) + 1).astype(int)
                wedge_ticks_right = 2 * int(np.floor(4 * M_horiz)) + 2 - wedge_ticks_left
                wedge_ticks = np.concatenate([wedge_ticks_left, wedge_ticks_right[-2::-1]])
            
            wedge_endpoints = wedge_ticks[1::2]  # 2:2:(end-1)
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2
            
            # Left corner wedge
            l += 1
            if l - 1 >= len(C[j_idx]):
                continue
            first_wedge_endpoint_vert = round(2 * int(np.floor(4 * M_vert)) / (2 * nbangles_perquad) + 1)
            length_corner_wedge = int(np.floor(4 * M_vert)) - int(np.floor(M_vert)) + int(np.ceil(first_wedge_endpoint_vert / 4))
            Y_corner = np.arange(1, length_corner_wedge + 1)
            XX, YY = np.meshgrid(np.arange(1, 2 * int(np.floor(4 * M_horiz)) + 2), Y_corner)
            width_wedge = wedge_endpoints[1] + wedge_endpoints[0] - 1
            slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[0]) / int(np.floor(4 * M_vert))
            left_line = np.round(2 - wedge_endpoints[0] + slope_wedge * (Y_corner - 1)).astype(int)
            
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            first_row = int(np.floor(4 * M_vert)) + 2 - int(np.ceil((length_corner_wedge + 1) / 2)) + \
                       ((length_corner_wedge + 1) % 2) * (quadrant - 2 == (quadrant - 2) % 2)
            first_col = int(np.floor(4 * M_horiz)) + 2 - int(np.ceil((width_wedge + 1) / 2)) + \
                       ((width_wedge + 1) % 2) * (quadrant - 3 == (quadrant - 3) % 2)
            
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                new_row = 1 + (row - first_row) % length_corner_wedge
                admissible_cols = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
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
                wrapped_data = fftshift(fft2(ifftshift(C[j_idx][l - 1]))) / np.sqrt(C[j_idx][l - 1].size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                if l - 1 + nbangles[j_idx] // 2 < len(C[j_idx]):
                    x = C[j_idx][l - 1] + 1j * C[j_idx][l - 1 + nbangles[j_idx] // 2]
                    wrapped_data = fftshift(fft2(ifftshift(x))) / np.sqrt(x.size) / np.sqrt(2)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    continue
            
            # Ensure window functions match wrapped_data shape
            if wl_left.shape != wrapped_data.shape:
                from scipy.ndimage import zoom
                zoom_factor_0 = wrapped_data.shape[0] / wl_left.shape[0]
                zoom_factor_1 = wrapped_data.shape[1] / wl_left.shape[1]
                wl_left = zoom(wl_left, (zoom_factor_0, zoom_factor_1), order=1)
                wr_right = zoom(wr_right, (zoom_factor_0, zoom_factor_1), order=1)
            
            wrapped_data = wrapped_data * (wl_left * wr_right)
            
            # Unwrapping data
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = np.round(0.5 * (cols + 1 + np.abs(cols - 1))).astype(int)
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
            
            for subl in range(2, nbangles_perquad):  # 2:(nbangles_perquad-1)
                l += 1
                if l - 1 >= len(C[j_idx]):
                    continue
                width_wedge = wedge_endpoints[subl] - wedge_endpoints[subl - 2] + 1
                slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[subl - 1]) / int(np.floor(4 * M_vert))
                left_line = np.round(wedge_endpoints[subl - 2] + slope_wedge * (Y - 1)).astype(int)
                
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
                    wrapped_data = fftshift(fft2(ifftshift(C[j_idx][l - 1]))) / np.sqrt(C[j_idx][l - 1].size)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    x = C[j_idx][l - 1] + 1j * C[j_idx][l - 1 + nbangles[j_idx] // 2]
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
            l += 1
            if l - 1 >= len(C[j_idx]):
                continue
            width_wedge = 4 * int(np.floor(4 * M_horiz)) + 3 - wedge_endpoints[-1] - wedge_endpoints[-2]
            slope_wedge = (int(np.floor(4 * M_horiz)) + 1 - wedge_endpoints[-1]) / int(np.floor(4 * M_vert))
            left_line = np.round(wedge_endpoints[-2] + slope_wedge * (Y_corner - 1)).astype(int)
            
            wrapped_XX = np.zeros((length_corner_wedge, width_wedge))
            wrapped_YY = np.zeros((length_corner_wedge, width_wedge))
            first_row = int(np.floor(4 * M_vert)) + 2 - int(np.ceil((length_corner_wedge + 1) / 2)) + \
                       ((length_corner_wedge + 1) % 2) * (quadrant - 2 == (quadrant - 2) % 2)
            first_col = int(np.floor(4 * M_horiz)) + 2 - int(np.ceil((width_wedge + 1) / 2)) + \
                       ((width_wedge + 1) % 2) * (quadrant - 3 == (quadrant - 3) % 2)
            
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = np.round(0.5 * (cols + 2 * int(np.floor(4 * M_horiz)) + 1 - 
                                                 np.abs(cols - (2 * int(np.floor(4 * M_horiz)) + 1)))).astype(int)
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
                wrapped_data = fftshift(fft2(ifftshift(C[j_idx][l - 1]))) / np.sqrt(C[j_idx][l - 1].size)
                wrapped_data = np.rot90(wrapped_data, quadrant - 1)
            else:
                if l - 1 + nbangles[j_idx] // 2 < len(C[j_idx]):
                    x = C[j_idx][l - 1] + 1j * C[j_idx][l - 1 + nbangles[j_idx] // 2]
                    wrapped_data = fftshift(fft2(ifftshift(x))) / np.sqrt(x.size) / np.sqrt(2)
                    wrapped_data = np.rot90(wrapped_data, quadrant - 1)
                else:
                    continue
            
            # Ensure window functions match wrapped_data shape
            if wl_left.shape != wrapped_data.shape:
                from scipy.ndimage import zoom
                zoom_factor_0 = wrapped_data.shape[0] / wl_left.shape[0]
                zoom_factor_1 = wrapped_data.shape[1] / wl_left.shape[1]
                wl_left = zoom(wl_left, (zoom_factor_0, zoom_factor_1), order=1)
                wr_right = zoom(wr_right, (zoom_factor_0, zoom_factor_1), order=1)
            
            wrapped_data = wrapped_data * (wl_left * wr_right)
            
            # Unwrapping data - MATLAB line 281: Xj(row,fliplr(admissible_cols)) = Xj(row,fliplr(admissible_cols)) + wrapped_data(new_row,end:-1:1);
            for row_idx, row in enumerate(Y_corner):
                cols = left_line[row_idx] + np.mod(np.arange(width_wedge) - (left_line[row_idx] - first_col), width_wedge)
                admissible_cols = np.round(0.5 * (cols + 2 * int(np.floor(4 * M_horiz)) + 1 - 
                                                 np.abs(cols - (2 * int(np.floor(4 * M_horiz)) + 1)))).astype(int)
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
    Xj = fftshift(fft2(ifftshift(C[0][0]))) / np.sqrt(C[0][0].size)
    loc_1 = Xj_topleft_1 + np.arange(2 * int(np.floor(4 * M1)) + 1) - 1
    loc_2 = Xj_topleft_2 + np.arange(2 * int(np.floor(4 * M2)) + 1) - 1
    X[np.ix_(loc_1, loc_2)] = X[np.ix_(loc_1, loc_2)] + Xj * lowpass
    
    # Finest level
    M1 = N1 / 3
    M2 = N2 / 3
    if finest == 1:
        # Folding back onto N1-by-N2 matrix
        shift_1 = int(np.floor(2 * M1)) - int(np.floor(N1 / 2))
        shift_2 = int(np.floor(2 * M2)) - int(np.floor(N2 / 2))
        
        # MATLAB: Y = X(:,(1:N2)+shift_2);
        Y = X[:, shift_2:shift_2 + N2]
        Y[:, N2 - shift_2:N2] = Y[:, N2 - shift_2:N2] + X[:, :shift_2]
        Y[:, :shift_2] = Y[:, :shift_2] + X[:, N2 + shift_2:N2 + 2 * shift_2]
        X = Y[shift_1:shift_1 + N1, :]
        X[N1 - shift_1:N1, :] = X[N1 - shift_1:N1, :] + Y[:shift_1, :]
        X[:shift_1, :] = X[:shift_1, :] + Y[N1 + shift_1:N1 + 2 * shift_1, :]
    else:
        # Extension to a N1-by-N2 matrix
        Y = fftshift(fft2(ifftshift(C[nbscales - 1][0]))) / np.sqrt(C[nbscales - 1][0].size)
        # Fix indexing for 0-based Python vs 1-based MATLAB
        # In 0-based indexing, center is N1//2, N2//2
        X_topleft_1 = N1 // 2 - int(np.floor(M1))
        X_topleft_2 = N2 // 2 - int(np.floor(M2))
        loc_1 = X_topleft_1 + np.arange(2 * int(np.floor(M1)) + 1)
        loc_2 = X_topleft_2 + np.arange(2 * int(np.floor(M2)) + 1)
        Y[np.ix_(loc_1, loc_2)] = Y[np.ix_(loc_1, loc_2)] * hipass_finest + X
        X = Y
    
    x = fftshift(ifft2(ifftshift(X))) * np.sqrt(X.size)
    if is_real:
        x = np.real(x)
    
    return x