import numpy as np
import glob
from astropy.io import fits
import sys
import os
import json
import argparse

# IMPORTANT - this is the substring that will be used to identify altered frames.
# In order to avoid recursive behaviour of altering already altered frames.
ALTERED_FRAME_SUBSTRING = "_alt_"


class FrameStack:

    def __init__(self, dir_path, filename, N_out, x_offset, y_offset, hdul_index):

        self.dir_path = dir_path
        self.filename = filename
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.hdul_index = hdul_index
        self.hdul = fits.open(os.path.join(dir_path, filename))

        self.header = self.hdul[hdul_index].header
        self.data = self.hdul[hdul_index].data
        self.N_out = N_out

        # these are to keep track of the shape of the data as it goes
        # through the different translations.
        self.x_shape = self.data.shape[1]
        self.y_shape = self.data.shape[0]

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
        }

        log_filename = os.path.join(
            self.dir_path, self.filename.split(".fits")[0] + "_log.json"
        )
        with open(log_filename, "w") as log_file:
            json.dump(log_data, log_file, indent=4)
        print(f"Log written to {log_filename}")

    def translate_x(self):

        if self.x_offset is None:
            print("No offset provided for x translation. Skipping...")
            return

        if self.x_offset == 0:
            print("Zero offset provided for x translation. Skipping...")
            return

        offsets = np.arange(0, self.N_out, self.x_offset)

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

        if self.y_offset is None:
            print("No offset provided for y translation. Skipping...")
            return

        if self.y_offset == 0:
            print("Zero offset provided for y translation. Skipping...")
            return

        offsets = np.arange(0, self.N_out, self.y_offset)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate frames that are subsequently offset from the input."
    )
    parser.add_argument(
        "dir", nargs="?", default=".", help="Directory containing the input FITS files."
    )
    parser.add_argument("N_out", type=int, help="Number of output frames to create")
    parser.add_argument(
        "--x_offset", type=int, default=0, help="Offset for x translation"
    )
    parser.add_argument(
        "--y_offset", type=int, default=0, help="Offset for y translation"
    )
    parser.add_argument(
        "--hdul_index", type=int, default=1, help="HDU index to read/write (default: 1)"
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

    y_offset = args.y_offset
    if y_offset < 0:
        print("y_offset must be a non-negative integer.")
        sys.exit(1)

    hdul_index = args.hdul_index
    if hdul_index < 0:
        print("hdul_index must be a non-negative integer.")
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

        fs = FrameStack(dir_path, filename, N_out, x_offset, y_offset, hdul_index)

        fs.translate_x()
        fs.translate_y()

        fs.write_log_to_file()

        fs.write_frames_to_fits()

        # we might end up working with large data cubes, so some forced garbage collection here.
        del fs
