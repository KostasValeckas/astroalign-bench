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
from utils import (
    shift_image,
    rotate_image,
    translate_x_wcs,
    translate_y_wcs,
    rotate_wcs,
)


def parse_number(value):
    """Argparse `type` that returns an int if the input is an integer string,
    otherwise returns a float. Raises `argparse.ArgumentTypeError` on failure.
    """
    try:
        # Try integer first so '1' becomes int, '1.0' will fall through
        iv = int(value)
        # Reject inputs like '1.0' which int() would accept by truncation if value is float string? int('1.0') raises, so safe.
        return iv
    except ValueError:
        try:
            fv = float(value)
            return fv
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid numeric value: {value}")


# IMPORTANT - this is the substring that will be used to identify altered frames.
# In order to avoid recursive behaviour of altering already altered frames.
ALTERED_FRAME_SUBSTRING = "_alt_"


class FrameStack:

    def __init__(
        self,
        dir_path,
        filename,
        N_out,
        x_offset,
        y_offset,
        hdul_index,
        gaussian_blur_fwhm=None,
        rot_angle=0.0,
    ):

        self.dir_path = dir_path
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.x_offsets = (
            np.arange(0, N_out * x_offset, x_offset)
            if x_offset != 0
            else np.zeros(N_out)
        )

        self.y_offsets = (
            np.arange(0, N_out * y_offset, y_offset)
            if y_offset != 0
            else np.zeros(N_out)
        )

        # Convert FWHM to sigma
        if gaussian_blur_fwhm is not None:
            self.gaussian_blur_sigma = gaussian_blur_fwhm / (2 * np.sqrt(2 * np.log(2)))

        self.hdul_index = hdul_index
        self.hdul = fits.open(os.path.join(dir_path, filename))

        self.header = self.hdul[hdul_index].header
        self.data = self.hdul[hdul_index].data
        self.N_out = N_out

        # these are to keep track of the shape of the data as it goes
        # through the different translations.
        self.x_shape = self.data.shape[1]
        self.y_shape = self.data.shape[0]

        self.rot_angle = rot_angle

        self.data_cube = np.stack([self.data] * self.N_out, axis=0)

        self.out_filenames = [
            self.filename.split(".fits")[0] + f"{ALTERED_FRAME_SUBSTRING}{i+1:03d}.fits"
            for i in range(self.N_out)
        ]

    def write_frames_to_fits(self):

        for i in range(self.N_out):
            new_name = os.path.join(self.dir_path, self.out_filenames[i])
            # Preserve the original HDU structure: copy each HDU from the opened input
            new_hdul = fits.HDUList([hdu.copy() for hdu in self.hdul])

            # Replace the data in the specified HDU index with the frame data
            # Ensure we don't modify the original opened HDU objects
            new_hdul[self.hdul_index].data = self.data_cube[i]

            # It's good practice to update any relevant header keywords if needed
            # (e.g., update DATE or HISTORY). Here we preserve existing headers.
            new_hdul.writeto(new_name, overwrite=True)
            print(f"Saved frame {i+1} to {new_name}")

    def delete_fits_files(self):
        for filename in self.out_filenames:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted {filename}")
            else:
                print(f"File {filename} does not exist, cannot delete.")

    def write_log_to_file(self):

        log_data = {
            "dir_path": self.dir_path,
            "filename": self.filename,
            "N_out": self.N_out,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "hdul_index": self.hdul_index,
            "out_filenames": self.out_filenames,
            "gaussian_blur_sigma": (
                self.gaussian_blur_sigma if hasattr(self, "gaussian_blur_sigma") else 0
            ),
            "rot_angle": self.rot_angle,
        }

        log_filename = os.path.join(
            self.dir_path, self.filename.split(".fits")[0] + "_log.json"
        )
        with open(log_filename, "w") as log_file:
            json.dump(log_data, log_file, indent=4)
        print(f"Log written to {log_filename}")

    def translate_x(self):
        """
        For whole pixel translations.
        """

        if self.x_offset is None:
            print("No offset provided for x translation. Skipping...")
            return

        if self.x_offset == 0:
            print("Zero offset provided for x translation. Skipping...")
            return

        offsets = self.x_offsets

        print(f"Translating image in x direction with offsets: {offsets}")

        _, x_size = self.data.shape

        max_offset = offsets[-1]

        # the shift will mutate the xsize - calulate the new xsize, take a
        # copy and recreate the data_cube with the new shape to add the modified frames.
        self.x_shape = x_size - max_offset

        cube_copy = self.data_cube.copy()

        self.data_cube = np.zeros((self.N_out, self.y_shape, self.x_shape))

        # Create offsets by cropping the image in x direction
        for i, offset in enumerate(offsets):
            if offset > 0:
                cropped_image = cube_copy[i, :, offset : x_size - (max_offset - offset)]
            else:
                cropped_image = cube_copy[i, :, : x_size - max_offset]

            self.data_cube[i] = cropped_image

    def translate_y(self):
        """
        For whole pixel translations.
        """

        if self.y_offset is None:
            print("No offset provided for y translation. Skipping...")
            return

        if self.y_offset == 0:
            print("Zero offset provided for y translation. Skipping...")
            return

        offsets = self.y_offsets

        print(f"Translating image in y direction with offsets: {offsets}")

        y_size, _ = self.data.shape

        max_offset = offsets[-1]

        # the shift will mutate the ysize - calculate the new ysize, take a
        # copy and recreate the data_cube with the new shape to add the modified frames.
        self.y_shape = y_size - max_offset

        cube_copy = self.data_cube.copy()

        self.data_cube = np.zeros((self.N_out, self.y_shape, self.x_shape))

        # Create offsets by cropping the image in y direction
        for i, offset in enumerate(offsets):
            if offset > 0:
                cropped_image = cube_copy[i, offset : y_size - (max_offset - offset), :]
            else:
                cropped_image = cube_copy[i, : y_size - max_offset, :]
            self.data_cube[i] = cropped_image

        return

    def shift_x_y(self):

        max_offset_x = self.x_offsets[-1]
        max_offset_y = self.y_offsets[-1]

        cropped_lower_x = int(max_offset_x)
        cropped_upper_x = int(self.x_shape - (max_offset_x - cropped_lower_x))

        cropped_lower_y = int(max_offset_y)
        cropped_upper_y = int(self.y_shape - (max_offset_y - cropped_lower_y))

        cube_copy = self.data_cube.copy()

        self.data_cube = np.zeros(
            (
                self.N_out,
                cropped_upper_y - cropped_lower_y,
                cropped_upper_x - cropped_lower_x,
            )
        )

        for i, image in enumerate(cube_copy):

            # to stick with the convention of positive offsets being in the positive direction
            dx = -self.x_offsets[i]
            dy = -self.y_offsets[i]

            x0 = cropped_lower_x
            x1 = cropped_upper_x

            y0 = cropped_lower_y
            y1 = cropped_upper_y

            shifted = shift_image(image, dx, dy)

            cropped = shifted[y0:y1, x0:x1]

            self.data_cube[i] = cropped

    def convolve_psf(self):
        print(f"Applying Gaussian PSF blurring with sigma: {self.gaussian_blur_sigma}")

        for i, image in enumerate(self.data_cube):

            if i == 0:
                # first image will be the referrence, don't blurr it
                continue

            self.data_cube[i] = gaussian_filter(image, sigma=self.gaussian_blur_sigma)

    def rotate_images(self):
        if self.rot_angle == 0.0:
            print("No rotation angle provided. Skipping rotation...")
            return

        print(f"Rotating images by {self.rot_angle} degrees.")

        for i in range(self.N_out):

            if i == 0:
                # first image will be the reference, don't rotate it
                continue

            image = self.data_cube[i]

            rotated = rotate_image(image, self.rot_angle)

            self.data_cube[i] = rotated

    def reproject_wcs(self):
        """
        Reproject the images using WCS transformations.
        """

        print("Reprojecting images using WCS transformations.")

        cube_copy = self.data_cube.copy()

        original_wcs = WCS(self.hdul[self.hdul_index].header).deepcopy()

        for i, frame in enumerate(cube_copy):

            if i == 0:
                # first image will be the reference, don't reproject it
                continue

            dx = self.x_offsets[i]
            dy = self.y_offsets[i]

            # Create a copy of the original WCS to avoid modifying it
            new_wcs = WCS(self.hdul[self.hdul_index].header).deepcopy()

            # Apply translation and rotation to the WCS
            new_wcs = translate_x_wcs(new_wcs, -dx)
            new_wcs = translate_y_wcs(new_wcs, -dy)
            new_wcs = rotate_wcs(new_wcs, self.x_shape, self.y_shape, self.rot_angle)

            misaligned, _ = reproject_interp(
                (frame, new_wcs), original_wcs, shape_out=(self.y_shape, self.x_shape)
            )

            self.data_cube[i] = misaligned


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate frames that are subsequently offset from the input."
    )
    parser.add_argument(
        "dir", nargs="?", default=".", help="Directory containing the input FITS files."
    )
    parser.add_argument("N_out", type=int, help="Number of output frames to create")
    parser.add_argument(
        "--x_offset",
        default=0,
        type=parse_number,
        help="Offset for x translation (int for whole-pixel, float for sub-pixel)",
    )
    parser.add_argument(
        "--y_offset",
        default=0,
        type=parse_number,
        help="Offset for y translation (int for whole-pixel, float for sub-pixel)",
    )

    parser.add_argument(
        "--blurr_fwhm",
        type=float,
        default=None,
        help="FWHM for Gaussian blurring (optional)",
    )

    parser.add_argument(
        "--hdul_index", type=int, default=1, help="HDU index to read/write (default: 1)"
    )

    parser.add_argument(
        "--rot_angle",
        type=float,
        default=0.0,
        help="Rotation angle in degrees (optional)",
    )

    parser.add_argument(
        "--skip_wcs_reprojection",
        action="store_true",
        help="Skip WCS reprojection (direct image manipulation) (default: False)",
    )

    args = parser.parse_args()

    dir_path = args.dir

    if not os.path.isdir(dir_path):
        print(f"Directory {dir_path} does not exist.")
        sys.exit(1)

    N_out = args.N_out

    if N_out <= 0:
        print("N_out must be a positive integer.")
        sys.exit(1)

    if N_out > 99:
        print("N_out must be less than or equal to 99.")
        sys.exit(1)

    x_offset = args.x_offset
    if x_offset < 0:
        print("x_offset must be a non-negative integer.")
        sys.exit(1)

    type_x = type(x_offset)

    y_offset = args.y_offset
    if y_offset < 0:
        print("y_offset must be a non-negative integer.")
        sys.exit(1)

    type_y = type(y_offset)

    hdul_index = args.hdul_index
    if hdul_index < 0:
        print("hdul_index must be a non-negative integer.")
        sys.exit(1)

    blurr_fwhm = args.blurr_fwhm
    if blurr_fwhm is not None and blurr_fwhm <= 0:
        print("blurr_fwhm must be a positive number.")
        sys.exit(1)

    # find first .fits file in directory
    fits_files = glob.glob(os.path.join(dir_path, "*.fits"))

    if not fits_files:
        print(f"No .fits files found in {dir_path}")
        sys.exit(1)

    for i, file in enumerate(fits_files):

        if ALTERED_FRAME_SUBSTRING in file:
            print(f"Skipping altered frame: {file}")
            continue  # skip already altered frames

        print(f"Found FITS file: {file}")

        filename = os.path.basename(fits_files[i])

        fs = FrameStack(
            dir_path,
            filename,
            N_out,
            x_offset,
            y_offset,
            hdul_index,
            gaussian_blur_fwhm=blurr_fwhm,
            rot_angle=args.rot_angle,
        )

        if args.skip_wcs_reprojection:
            if (type_x is int) and (type_y is int):
                fs.translate_x()
                fs.translate_y()

            # sub pixel shift
            else:
                fs.shift_x_y()
                fs.rotate_images()

        else:
            print("Using WCS reprojection for image transformations.")
            fs.reproject_wcs()

        if blurr_fwhm is not None:
            fs.convolve_psf()

        fs.write_log_to_file()

        fs.write_frames_to_fits()

        # we might end up working with large data cubes, so some forced garbage collection here.
        del fs
