import numpy as np


def fdct_wrapping_window(x):
    """
    fdct_wrapping_window - Creates the two halves of a C^inf compactly supported window
    
    Inputs:
        x : array_like
            Vector or matrix of abscissae, the relevant ones from 0 to 1
    
    Outputs:
        wl, wr : ndarray
            Arrays containing samples of the left and right half of the window
    
    This is a direct Python port of fdct_wrapping_window.m by Laurent Demanet, 2004
    """
    x = np.asarray(x)
    
    # Initialize output arrays
    wr = np.zeros_like(x, dtype=float)
    wl = np.zeros_like(x, dtype=float)
    
    # Set very small values to zero (MATLAB uses 2^-52)
    x = np.where(np.abs(x) < 2**-52, 0, x)
    
    # Right window: wr((x > 0) & (x < 1)) = exp(1-1./(1-exp(1-1./x((x > 0) & (x < 1)))));
    mask_right = (x > 0) & (x < 1)
    if np.any(mask_right):
        x_vals = x[mask_right]
        wr[mask_right] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / x_vals)))
    
    # wr(x <= 0) = 1;
    wr[x <= 0] = 1
    
    # Left window: wl((x > 0) & (x < 1)) = exp(1-1./(1-exp(1-1./(1-x((x > 0) & (x < 1))))));
    mask_left = (x > 0) & (x < 1)
    if np.any(mask_left):
        x_vals = 1 - x[mask_left]  # Note: 1-x
        wl[mask_left] = np.exp(1 - 1 / (1 - np.exp(1 - 1 / x_vals)))
    
    # wl(x >= 1) = 1;
    wl[x >= 1] = 1
    
    # Normalization
    normalization = np.sqrt(wl**2 + wr**2)
    
    # Avoid division by zero
    mask_nonzero = normalization != 0
    wr = np.where(mask_nonzero, wr / normalization, 0)
    wl = np.where(mask_nonzero, wl / normalization, 0)
    
    return wl, wr