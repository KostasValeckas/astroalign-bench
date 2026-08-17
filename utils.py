import numpy as np
from scipy.ndimage import rotate, shift
from astropy.wcs import WCS

"""
A helper module to avoid repeated code.
"""

#TODO the wcs methods don't need to be here, could be in FrameStack class
def translate_x_wcs(wcs, dx):

    new_wcs = wcs.deepcopy()

    new_wcs.wcs.crpix[0] -= dx

    return new_wcs

def translate_y_wcs(wcs, dy):

    new_wcs = wcs.deepcopy()

    new_wcs.wcs.crpix[1] -= dy

    return new_wcs

def rotate_wcs(wcs, nx, ny, theta):

    new_wcs = wcs.deepcopy()

    theta = np.deg2rad(theta)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Image center
    center = np.array([
        (nx - 1) / 2,
        (ny - 1) / 2
    ])

    # Original reference pixel
    crpix = np.array(new_wcs.wcs.crpix)

    # Rotate CRPIX around image center
    new_crpix = center + R @ (crpix - center)

    new_wcs.wcs.crpix = new_crpix

    # Rotate the WCS orientation
    new_wcs.wcs.cd = new_wcs.wcs.cd @ R.T

    return new_wcs

def shift_image(image, dx, dy):
    """
    Shift the image by the given offsets.

    Takes care of some NaN values that may appear in the image after shifting.
    """

    valid = np.isfinite(image)

    image_filled = np.nan_to_num(image, nan=0.0)

    shifted = shift(image_filled, shift=(dy, dx), order=3, mode="constant", cval=0.0)

    shifted_valid = shift(
        valid.astype(float), shift=(dy, dx), order=1, mode="constant", cval=0.0
    )

    shifted[shifted_valid < 0.5] = np.nan

    return shifted

def rotate_image(image, angle):
    """
    Rotate the image by the given angle.

    Takes care of some NaN values that may appear in the image after rotation.
    """

    # 1. Where was the original image actually observed?
    valid = np.isfinite(image)

    # 2. Replace NaNs with zero temporarily
    image_filled = np.nan_to_num(image, nan=0.0)

    # 3. Rotate the flux
    rotated = rotate(
        image_filled,
        angle=angle,
        reshape=False,
        order=3,
        mode="constant",
        cval=0.0
    )
    
    # 4. Rotate the validity map
    rotated_valid = rotate(
        valid.astype(float),
        angle=angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0
    )
    
    # 5. Restore NaNs where there isn't valid data
    rotated[rotated_valid < 0.5] = np.nan

    return rotated
