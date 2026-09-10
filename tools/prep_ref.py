import numpy as np
import glob
from astropy.io import fits
from reproject import reproject_interp
import sys
import os
import json
import argparse
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils import (
    shift_image,
    rotate_image,
    translate_x_wcs,
    translate_y_wcs,
    rotate_wcs,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uses WCS to reproject the reference image to match the target image."
    )
    parser.add_argument(
        "ref_image",
        type=str,
        help="Path to the reference image FITS file",
    )
    parser.add_argument(
        "target_image",
        type=str,
        help="Path to the target image FITS file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=".",
        help="Directory to save the processed reference image",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        required=False,
        default="processed_ref_image.fits",
        help="Name of the output FITS file for the processed reference image",
    )

    parser.add_argument(
        "--hdu_index_ref",
        type=int,
        default=0,
        help="HDU index for the reference image (default: 0)",
    )

    parser.add_argument(
        "--hdu_index_target",
        type=int,
        default=1,
        help="HDU index for the target image (default: 1)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the reprojected reference image (default: False)",
    )



    args = parser.parse_args()

    # Load reference and target images
    ref_hdu = fits.open(args.ref_image)[args.hdu_index_ref]
    target_hdu = fits.open(args.target_image)[args.hdu_index_target]

    ref_data = ref_hdu.data
    ref_header = ref_hdu.header

    target_data = target_hdu.data
    target_header = target_hdu.header

    # Reproject the reference image to match the target image
    reprojected_data, _ = reproject_interp((ref_data, ref_header), target_header)

    if args.show:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(reprojected_data, origin="lower", cmap="gray")
        ax1.set_title("Reprojected Reference Image")
        ax2.imshow(target_data, origin="lower", cmap="gray")
        ax2.set_title("Target Image")
        plt.show()

    # Save the reprojected reference image
    output_path = os.path.join(args.output_dir, args.output_name)
    fits.writeto(output_path, reprojected_data, target_header, overwrite=True)





