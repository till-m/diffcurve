'''2D discrete curvelet transform in numpy'''
import numpy as np
from numpy import fft


def perform_fft2(spatial_input: np.ndarray):
    """Perform fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """

    return fft.fftshift(fft.fft2(fft.ifftshift(spatial_input), norm='ortho'))


def perform_ifft2(frequency_input: np.ndarray):
    """Perform inverse fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return fft.fftshift(fft.ifft2(fft.ifftshift(frequency_input),
                                  norm='ortho'))


def numpy_fdct_2d(img, curvelet_system, order='unshifted'):
    """2d fast discrete curvelet in numpy

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain
        order: 'unshifted' (fastest, default), 'frequency', or 'shifted'

    Returns:
        coeffs: curvelet coefficients
    """
    if order == 'unshifted':
        # Optimized: work in unshifted frequency domain (2 shifts total)
        x_freq_unshifted = fft.fft2(fft.ifftshift(img), norm='ortho')
        coeffs_freq_unshifted = x_freq_unshifted * np.conj(curvelet_system)
        coeffs = fft.fftshift(fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        x_freq = perform_fft2(img)
        coeffs = perform_ifft2(x_freq * np.conj(curvelet_system))
        return coeffs
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        x_freq_unshifted = fft.fft2(fft.ifftshift(img), norm='ortho')
        curvelet_system_unshifted = fft.ifftshift(curvelet_system, axes=(-2, -1))
        coeffs_freq_unshifted = x_freq_unshifted * np.conj(curvelet_system_unshifted)
        coeffs = fft.fftshift(fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")


def numpy_ifdct_2d(coeffs, curvelet_system, curvelet_support_size, order='unshifted'):
    """2d inverse fast discrete curvelet in numpy

    Args:
        coeffs: curvelet coefficients
        curvelet_system: curvelet waveforms in the frequency domain
        curvelet_support_size: size of the support of each curvelet wedge
        order: 'unshifted' (fastest, default), 'frequency', or 'shifted'

    Returns:
        decom: image decomposed in different scales and orientation in the
        curvelet basis.
    """
    if order == 'unshifted':
        # Optimized: work in unshifted frequency domain (2 shifts total)
        coeffs_freq_unshifted = fft.fft2(fft.ifftshift(coeffs), norm='ortho')
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system
        decomp = fft.fftshift(fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size[..., np.newaxis, np.newaxis]
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        coeffs_freq = perform_fft2(coeffs)
        decomp = perform_ifft2(coeffs_freq * curvelet_system)
        return decomp * curvelet_support_size[..., np.newaxis, np.newaxis]
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        coeffs_freq_unshifted = fft.fft2(fft.ifftshift(coeffs), norm='ortho')
        curvelet_system_unshifted = fft.ifftshift(curvelet_system, axes=(-2, -1))
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system_unshifted
        decomp = fft.fftshift(fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size[..., np.newaxis, np.newaxis]
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")
