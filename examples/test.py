import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from diffcurve.fdct2d import get_curvelet_system
from diffcurve.fdct2d import fdct_wrapping, ifdct_wrapping
from diffcurve.utils import get_project_root

project_root = get_project_root()

lena_file = Path.joinpath(project_root, "data/Lena.jpg")

lena_img_square_even = cv2.imread(str(lena_file), 0).astype(float) / 255

def get_curvelet_decomposition(img, dct_kwargs):
    zeros = np.zeros_like(img)
    zero_coeffs = fdct_wrapping(zeros,
                                dct_kwargs['is_real'],
                                dct_kwargs['finest'],
                                dct_kwargs['nbscales'],
                                dct_kwargs['nbangles_coarse'])

    tmp = fdct_wrapping(zeros,
                            dct_kwargs['is_real'],
                            dct_kwargs['finest'],
                            dct_kwargs['nbscales'],
                            dct_kwargs['nbangles_coarse'])

    img_coeffs = fdct_wrapping(img,
                                   dct_kwargs['is_real'],
                                   dct_kwargs['finest'],
                                   dct_kwargs['nbscales'],
                                   dct_kwargs['nbangles_coarse'])

    all_scales_all_wedges_curvelet_coeffs = []
    for (scale_idx, curvelets_scale) in enumerate(zero_coeffs):
        print(f'scale {scale_idx}')
        for (wedge_idx, curvelet_wedge) in  enumerate(curvelets_scale):

            tmp[scale_idx][wedge_idx] = img_coeffs[scale_idx][wedge_idx]
            out = np.array(ifdct_wrapping(tmp))
            all_scales_all_wedges_curvelet_coeffs.append(out)
            tmp[scale_idx][wedge_idx] = zero_coeffs[scale_idx][wedge_idx]
    return all_scales_all_wedges_curvelet_coeffs

img = lena_img_square_even

dct_kwargs = {
    'is_real': 0, # complex-valued curvelets
    'finest': 2, # use wavelets at the finest level
    'nbscales': 6,
    'nbangles_coarse': 16}

matlab_decomp = get_curvelet_decomposition(img, dct_kwargs)
matlab_decomp = np.array(matlab_decomp)

print(f'MSE = { np.mean( (matlab_decomp.sum(0).real - img) ** 2 ) }')
