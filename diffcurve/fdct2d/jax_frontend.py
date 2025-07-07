'''2D discrete curvelet transform in jax'''
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)


def jax_perform_fft2(spatial_input):
    """Perform fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(spatial_input),
                                         norm='ortho'))


def jax_perform_ifft2(frequency_input):
    """Perform inverse fast fourier transform in 2D.
    The ifftshift and fftshift shift the origin of the signal. See
    http://www.cvl.isy.liu.se/education/undergraduate/tsbb14/laborationerFiler/Lab3_student.pdf
    """
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(frequency_input),
                                          norm='ortho'))


def jax_fdct_2d(img, curvelet_system, order='unshifted'):
    """2d fast discrete curvelet in jax

    Args:
        img: 2D array
        curvelet_system: curvelet waveforms in the frequency domain
        order: 'unshifted' (fastest, default), 'frequency', or 'shifted'

    Returns:
        coeffs: curvelet coefficients
    """
    if order == 'unshifted':
        # Optimized: work in unshifted frequency domain (2 shifts total)
        x_freq_unshifted = jnp.fft.fft2(jnp.fft.ifftshift(img), norm='ortho')
        coeffs_freq_unshifted = x_freq_unshifted * jnp.conj(curvelet_system)
        coeffs = jnp.fft.fftshift(jnp.fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        x_freq = jax_perform_fft2(img)
        coeffs = jax_perform_ifft2(x_freq * jnp.conj(curvelet_system))
        return coeffs
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        x_freq_unshifted = jnp.fft.fft2(jnp.fft.ifftshift(img), norm='ortho')
        curvelet_system_unshifted = jnp.fft.ifftshift(curvelet_system, axes=(-2, -1))
        coeffs_freq_unshifted = x_freq_unshifted * jnp.conj(curvelet_system_unshifted)
        coeffs = jnp.fft.fftshift(jnp.fft.ifft2(coeffs_freq_unshifted, norm='ortho'))
        return coeffs
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")


def jax_ifdct_2d(coeffs, curvelet_system, curvelet_support_size, order='unshifted'):
    """2d inverse fast discrete curvelet in jax

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
        coeffs_freq_unshifted = jnp.fft.fft2(jnp.fft.ifftshift(coeffs), norm='ortho')
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system
        decomp = jnp.fft.fftshift(jnp.fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size[..., jnp.newaxis, jnp.newaxis]
    elif order == 'frequency':
        # Original implementation (4 shifts total)
        coeffs_freq = jax_perform_fft2(coeffs)
        decomp = jax_perform_ifft2(coeffs_freq * curvelet_system)
        return decomp * curvelet_support_size[..., jnp.newaxis, jnp.newaxis]
    elif order == 'shifted':
        # Work with pre-shifted curvelet system
        coeffs_freq_unshifted = jnp.fft.fft2(jnp.fft.ifftshift(coeffs), norm='ortho')
        curvelet_system_unshifted = jnp.fft.ifftshift(curvelet_system, axes=(-2, -1))
        decomp_freq_unshifted = coeffs_freq_unshifted * curvelet_system_unshifted
        decomp = jnp.fft.fftshift(jnp.fft.ifft2(decomp_freq_unshifted, norm='ortho'))
        return decomp * curvelet_support_size[..., jnp.newaxis, jnp.newaxis]
    else:
        raise ValueError(f"Unknown order '{order}'. Use 'unshifted', 'frequency', or 'shifted'.")
