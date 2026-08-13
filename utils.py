import numpy as np
from scipy.ndimage import rotate, shift

"""
A helper module to avoid repeated code.
"""


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
