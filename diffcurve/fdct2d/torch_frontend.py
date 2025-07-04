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


def torch_fdct_2d(img, curvelet_system):
    """2d fast discrete curvelet in torch

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain

    Returns:
        coeffs: curvelet coefficients
    """
    x_freq = torch_perform_fft2(img)
    conj_curvelet_system = torch.conj(curvelet_system)
    coeffs = torch_perform_ifft2(einops.einsum(x_freq, conj_curvelet_system, '... h w, c h w -> ... c h w'))

    return coeffs


def torch_ifdct_2d(coeffs, curvelet_system, curvelet_support_size):
    """2d inverse fast discrete curvelet in torch

    Args:
        coeffs: curvelet coefficients
        curvelet_system: curvelet waveforms in the frequency domain
        curvelet_support_size: size of the support of each curvelet wedge

    Returns:
        decom: image decomposed in different scales and orientation in the
        curvelet basis.
    """
    coeffs_freq = torch_perform_fft2(coeffs)
    decom = einops.einsum(coeffs_freq, curvelet_system, '... c h w, c h w -> ... h w')

    decom = torch_perform_ifft2(
        einops.einsum(coeffs_freq, curvelet_system, '... c h w, c h w -> ... c h w'))
    decom = einops.einsum(decom, curvelet_support_size, '... c h w, c -> ... c h w')
    return decom
