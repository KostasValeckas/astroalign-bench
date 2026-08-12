import glob

import numpy as np
import matplotlib.pyplot as plt
import json
from astropy.io import fits
import os
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import spacepylot.alignment as align
import spacepylot.plotting as pl
from spacepylot.alignment_utilities import TranslationTransform


class AlignmentMethod:
    """
    Parent class for defining API of a method being tested
    """

    def __init__(self, log_file_path, method_name):

        self.log_file_path = log_file_path
        self.method_name = method_name

        try:
            log_dir = json.load(open(log_file_path, "r"))

        except FileNotFoundError:
            print(f"Log file not found: {log_file_path}")
            raise

        except json.JSONDecodeError:
            print(f"Invalid JSON in log file: {log_file_path}")
            raise

        self.dir_path = log_dir["dir_path"]
        self.filename = log_dir["filename"]
        self.N_out = log_dir["N_out"]

        self.x_offset = log_dir["x_offset"]
        self.x_offsets = (
            np.arange(0, self.N_out, self.x_offset)
            if self.x_offset != 0
            else np.zeros(self.N_out)
        )

        self.y_offset = log_dir["y_offset"]
        self.y_offsets = (
            np.arange(0, self.N_out, self.y_offset)
            if self.y_offset != 0
            else np.zeros(self.N_out)
        )

        self.hdul_index = log_dir["hdul_index"]

        self.out_filenames = log_dir["out_filenames"]

        self.gaussian_blur_sigma = log_dir["gaussian_blur_sigma"]

        # defined when the method is run
        self.recovered_x_offsets = None
        self.recovered_y_offsets = None

        self.x_error = None
        self.y_error = None

    def run_method(self):
        """
        Abstract method to be implemented by subclasses.
        This method should run the alignment method and populate the recovered offsets.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class MusePipeline(AlignmentMethod):
    """
    Class for handling the ESO MUSE pipeline alignment method.
    """

    def __init__(self, log_file_path):
        super().__init__(log_file_path, "MUSE Pipeline")

        # these are needed to manipulate .sof file production

        self.FILES = ["OFFSET_LIST.fits", "PREVIEW_FOV.fits"]

        self.SOURCE_IMAGE_TUPLES = []

        for i in range(1, self.N_out + 1):
            self.FILES.append(f"SOURCE_LIST_{i:04d}.fits")
            self.SOURCE_IMAGE_TUPLES.append(f"SOURCE_LIST_{i:04d}.fits")

        self.sof_filename = f"{self.filename.split('.')[0]}.sof"

        # MUSE pipelines saves the offsets in degrees, these are used for
        # intermediate calculations
        self.dra = None
        self.ddec = None

    def generate_sof_file(self, file):

        with open(self.sof_filename, "w") as sof_file:
            sof_file.write(f"{self.out_filenames[0]}    IMAGE_FOV\n")
            sof_file.write(f"{file}    IMAGE_FOV\n")

        print(f"Generated .sof file: {self.sof_filename}")

    def run_method(self):

        """
        Run the MUSE pipeline alignment method.

        Only running one at a time, because if it fails for one frames the 
        whole method does not produce the OFFSET_LIST.fits file 
        and the rest of the frames cannot be processed.
        """

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []

        current_dir_backup = os.getcwd()

        for i, file in enumerate(self.out_filenames):

            
            # a bit hacky but to avoid weird file writing
            if i == 0: os.chdir(self.dir_path)

            self.generate_sof_file(file)

            command = f"esorex muse_exp_align {self.sof_filename}"

            os.system(command)


            # get all the degree offsets
            offset_file_path = f"./OFFSET_LIST.fits"

            # this is a bit hacky failsafe when MUSE pipeline fails
            try:
                self.dra = fits.getdata(offset_file_path)["RA_OFFSET"][-1]
                self.ddec = fits.getdata(offset_file_path)["DEC_OFFSET"][-1]
            except FileNotFoundError:
                self.recovered_x_offsets.append(0)
                self.recovered_y_offsets.append(0)
                continue
                

            file_path = f"./{file}"

            hdul = fits.open(file_path)

            wcs = WCS(hdul[self.hdul_index].header)

            ra_center = hdul[self.hdul_index].header["CRVAL1"]
            dec_center = hdul[self.hdul_index].header["CRVAL2"]

            # Reference sky position (the point where offset starts)
            ref = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame="icrs")

            # Sky offset
            dra = self.dra * u.deg
            ddec = self.ddec * u.deg

            target = SkyCoord(
                ra=ra_center * u.deg + dra, dec=dec_center * u.deg + ddec, frame="icrs"
            )

            # Convert both sky positions to pixels
            x0, y0 = wcs.world_to_pixel(ref)
            x1, y1 = wcs.world_to_pixel(target)

            # Pixel offset
            dx = x1 - x0
            dy = y1 - y0

            # Negative sign because the offsets are in the direction of
            # the shift
            self.recovered_x_offsets.append(-dx)
            self.recovered_y_offsets.append(-dy)


        print(f"Recovered x offsets: {self.recovered_x_offsets}")
        print(f"Recovered y offsets: {self.recovered_y_offsets}")

        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)

        # lastly, clean up intermediate files generated by the MUSE pipeline
        for file in self.FILES:
            file_path = f"./{file}"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            else:
                print(f"File {file_path} does not exist, cannot delete.")

        # and lastly some manual deletes
        os.remove(f"./esorex.log")
        for p in glob.glob("./*.sof"):
            try:
                os.remove(p)
            except OSError as e:
                print(f"Failed to remove {p}: {e}")

        # Change back to the original directory
        os.chdir(current_dir_backup)


class SpacePylot(AlignmentMethod):

    def __init__(self, log_file_path):
        super().__init__(log_file_path, "SpacePylot")

    def run_method(self):

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []

        path_reference = f"{self.dir_path}/{self.out_filenames[0]}"

        for _, file in enumerate(self.out_filenames):

            path_prealign = f"{self.dir_path}/{file}"

            op = align.AlignOpticalFlow.from_fits(
                path_prealign,
                path_reference,
                hdu_index_reference=self.hdul_index,
                hdu_index_prealign=self.hdul_index,
            )

            # solution
            op.get_iterate_translation_rotation(homography_method=TranslationTransform)

            self.recovered_x_offsets.append(op.translation[0])
            self.recovered_y_offsets.append(op.translation[1])

        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
