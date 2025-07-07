'''2D discrete curvelet transform in torch'''
import torch
import einops

def torch_perform_fft2(spatial_input):
    """Perform fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """

    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
        spatial_input), norm='ortho'))


def torch_perform_ifft2(frequency_input):
    """Perform inverse fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(
        frequency_input), norm='ortho'))


def torch_fdct_2d(img, curvelet_system, order='unshifted'):
    """2d fast discrete curvelet in torch

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain
        order: 'unshifted' (fastest, default), 'frequency', or 'shifted'

    Returns:
        coeffs: curvelet coefficients
    """
    if order == 'unshifted':
        # Optimized: work in unshifted frequency domain (2 shifts total)
        x_freq_unshifted = torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')
        coeffs_freq_unshifted = x_freq_unshifted.unsqueeze(-3) * torch.conj(curvelet_system)
        coeffs = torch.fft.fftshift(torch.fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        x_freq = torch_perform_fft2(img)
        coeffs = torch_perform_ifft2(einops.einsum(x_freq, torch.conj(curvelet_system), '... h w, c h w -> ... c h w'))
        return coeffs
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        x_freq_unshifted = torch.fft.fft2(torch.fft.ifftshift(img), norm='ortho')
        curvelet_system_unshifted = torch.fft.ifftshift(curvelet_system, dim=(-2, -1))
        coeffs_freq_unshifted = x_freq_unshifted.unsqueeze(-3) * torch.conj(curvelet_system_unshifted)
        coeffs = torch.fft.fftshift(torch.fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")


def torch_ifdct_2d(coeffs, curvelet_system, curvelet_support_size, order='unshifted'):
    """2d inverse fast discrete curvelet in torch

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
        coeffs_freq_unshifted = torch.fft.fft2(torch.fft.ifftshift(coeffs), norm='ortho')
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system
        decomp = torch.fft.fftshift(torch.fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size.unsqueeze(-1).unsqueeze(-1)
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        coeffs_freq = torch_perform_fft2(coeffs)
        decom = torch_perform_ifft2(einops.einsum(coeffs_freq, curvelet_system, '... c h w, c h w -> ... c h w'))
        return einops.einsum(decom, curvelet_support_size, '... c h w, c -> ... c h w')
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        coeffs_freq_unshifted = torch.fft.fft2(torch.fft.ifftshift(coeffs), norm='ortho')
        curvelet_system_unshifted = torch.fft.ifftshift(curvelet_system, dim=(-2, -1))
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system_unshifted
        decomp = torch.fft.fftshift(torch.fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size.unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")
