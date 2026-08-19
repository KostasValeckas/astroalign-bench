import glob

from matplotlib import image
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
from utils import rotate_image, shift_image
from scipy.optimize import least_squares
import sys
from image_registration import chi2_shift
from image_registration.fft_tools import shift as fft_shift
import subprocess
import signal
import cv2


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
            np.arange(0, self.N_out * self.x_offset, self.x_offset)
            if self.x_offset != 0
            else np.zeros(self.N_out)
        )

        self.y_offset = log_dir["y_offset"]
        self.y_offsets = (
            np.arange(0, self.N_out * self.y_offset, self.y_offset)
            if self.y_offset != 0
            else np.zeros(self.N_out)
        )

        self.rot_angle = log_dir["rot_angle"]

        self.hdul_index = log_dir["hdul_index"]

        self.out_filenames = log_dir["out_filenames"]

        self.gaussian_blur_sigma = log_dir["gaussian_blur_sigma"]

        # defined when the method is run
        self.recovered_x_offsets = None
        self.recovered_y_offsets = None
        self.recovered_angles = None

        self.x_error = None
        self.y_error = None
        self.angle_error = None

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

        self.local_working_dir = os.path.join(self.dir_path, self.filename.split(".")[0])

    def generate_sof_file(self, file):

        with open(self.sof_filename, "w") as sof_file:
            sof_file.write(f"../{self.out_filenames[0]}    IMAGE_FOV\n")
            sof_file.write(f"../{file}    IMAGE_FOV\n")

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
        self.recovered_angles = []

        current_dir_backup = os.getcwd()

        

        for i, file in enumerate(self.out_filenames):

            # a bit hacky but to avoid weird file writing
            if i == 0:
                if not os.path.exists(self.local_working_dir):
                    os.mkdir(self.local_working_dir)
                os.chdir(self.local_working_dir)

            self.generate_sof_file(file)

            p = subprocess.Popen(
                ["esorex", "muse_exp_align", self.sof_filename],
                start_new_session=True,
            )

            try:
                p.wait()

            except KeyboardInterrupt:
                print("Ctrl-C: terminating esorex", flush=True)

                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()

                raise
            # get all the degree offsets
            offset_file_path = f"./OFFSET_LIST.fits"

            # this is a bit hacky failsafe when MUSE pipeline fails
            try:
                self.dra = fits.getdata(offset_file_path)["RA_OFFSET"][-1]
                self.ddec = fits.getdata(offset_file_path)["DEC_OFFSET"][-1]
            except FileNotFoundError:
                self.recovered_x_offsets.append(0)
                self.recovered_y_offsets.append(0)
                # MUSE pipeline does not recover angles
                self.recovered_angles.append(0)
                continue

            file_path = f"./{file}"

            hdul = fits.open(f"../{file_path}")

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
            # MUSE pipeline does not recover angles
            self.recovered_angles.append(0)

        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.x_error = np.abs(self.x_error)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
        self.y_error = np.abs(self.y_error)
        self.angle_error = np.array(self.recovered_angles) - self.rot_angle
        self.angle_error = np.abs(self.angle_error)
        self.angle_error[0] = 0  # first frame is reference frame, so no angle error

        # lastly, clean up intermediate files generated by the MUSE pipeline
        for file in self.FILES:
            file_path = f"./{file}"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            else:
                print(f"File {file_path} does not exist, cannot delete.")

        # and lastly some manual deletes
        try:
            os.remove(f"./esorex.log")
        except FileNotFoundError as e:
            print(f"File esorex.log not found: skipping removal. Error: {e}...")

        try: 
            os.remove(f"./{self.sof_filename}")
        except FileNotFoundError as e:
            print(f"File {self.sof_filename} not found: skipping removal. Error: {e}...")

        # Change back to the original directory
        os.chdir(current_dir_backup)

        os.rmdir(self.local_working_dir)


class SpacePylot(AlignmentMethod):

    def __init__(self, log_file_path):
        super().__init__(log_file_path, "SpacePylot")


    def run_method(self):

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []
        self.recovered_angles = []

        path_reference = f"{self.dir_path}/{self.out_filenames[0]}"

        for _, file in enumerate(self.out_filenames):

            path_prealign = f"{self.dir_path}/{file}"

            # first calculate rotation, correct for it, and then
            # calculate translation offsets

            data_reference = fits.getdata(path_reference, self.hdul_index)
            data_prealign = fits.getdata(path_prealign, self.hdul_index)

            op = align.AlignOpticalFlow(
                data_prealign,
                data_reference,
            )

            # solution
            op.get_iterate_translation_rotation()

            # trying - first correct rotation and then calculate offsets

            self.recovered_angles.append(-op.rotation_deg)

            rotated_data = rotate_image(data_prealign, op.rotation_deg)

            # rerun now only for translation
            op = align.AlignOpticalFlow(
                rotated_data,
                data_reference,
            )

            op.get_iterate_translation_rotation(homography_method=TranslationTransform)

            self.recovered_x_offsets.append(op.translation[1])
            self.recovered_y_offsets.append(op.translation[0])

        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.x_error = np.abs(self.x_error)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
        self.y_error = np.abs(self.y_error)
        self.angle_error = np.array(self.recovered_angles) - self.rot_angle
        self.angle_error = np.abs(self.angle_error)
        self.angle_error[0] = 0  # first frame is reference frame, so no angle error

class MinimizeDifference(AlignmentMethod):

    def __init__(self, log_file_path):
        super().__init__(log_file_path, "MinimizeDifference")

        self.initial_guess = [0, 0, 0]  # dx, dy, angle
        self.bounds = ([-100, -100, -180], [100, 100, 180])  # bounds for dx, dy, angle

    def transform_function(self, dx, dy, angle, reference_image):


        # Rotate the target image by the given angle
        rotated_image = rotate_image(reference_image, angle)

        # Shift the rotated image by the given offsets
        shifted_image = shift_image(rotated_image, dx, dy)

        return shifted_image



    def residuals(self, params, reference_image, prealign_image):

        dx, dy, angle = params

        transformed_image = self.transform_function(dx, dy, angle, prealign_image)

        # Keep a fixed residual length for least_squares. Dropping NaNs here
        # makes the vector length change as the transform moves invalid pixels.
        valid_pixels = np.isfinite(reference_image) & np.isfinite(prealign_image)
        resids = transformed_image[valid_pixels] - reference_image[valid_pixels]

        return np.nan_to_num(resids, nan=0.0, posinf=0.0, neginf=0.0)

    

    def run_method(self):

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []
        self.recovered_angles = []

        path_reference = f"{self.dir_path}/{self.out_filenames[0]}"

        for _, file in enumerate(self.out_filenames):

            print(f"Running {self.method_name} on file: {file}")

            path_prealign = f"{self.dir_path}/{file}"

            data_reference = fits.getdata(path_reference, self.hdul_index)
            data_prealign = fits.getdata(path_prealign, self.hdul_index)



            # Use least squares optimization to find the best dx, dy, and angle
            result = least_squares(
                self.residuals,
                self.initial_guess,
                args=(data_reference, data_prealign),
                bounds=self.bounds
            )

            print(f"Recovered parameters for {file}: dx={result.x[0]}, dy={result.x[1]}, angle={result.x[2]}")

            self.recovered_x_offsets.append(result.x[0])
            self.recovered_y_offsets.append(result.x[1])
            self.recovered_angles.append(-result.x[2])

        self.angle_error = np.array(self.recovered_angles) - self.rot_angle
        self.angle_error = np.abs(self.angle_error)
        self.angle_error[0] = 0  # first frame is reference frame, so no angle error
        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.x_error = np.abs(self.x_error)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
        self.y_error = np.abs(self.y_error)


class Chi2Shift(AlignmentMethod):

    def __init__(self, log_file_path):
        super().__init__(log_file_path, "Chi2Shift")

    def run_method(self):

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []
        self.recovered_angles = []

        path_reference = f"{self.dir_path}/{self.out_filenames[0]}"

        for _, file in enumerate(self.out_filenames):

            print(f"Running {self.method_name} on file: {file}")

            path_prealign = f"{self.dir_path}/{file}"

            data_reference = fits.getdata(path_reference, self.hdul_index)
            data_prealign = fits.getdata(path_prealign, self.hdul_index)

            dx, dy = chi2_shift(data_reference, data_prealign,return_error=False, upsample_factor='auto')

            print(f"Recovered parameters for {file}: dx={dx}, dy={dy}")

            self.recovered_x_offsets.append(-dx)
            self.recovered_y_offsets.append(-dy)
            self.recovered_angles.append(0)  # Chi2Shift does not recover angles

        self.angle_error = np.array(self.recovered_angles) - self.rot_angle
        self.angle_error = np.abs(self.angle_error)
        self.angle_error[0] = 0  # first frame is reference frame, so no angle error
        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.x_error = np.abs(self.x_error)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
        self.y_error = np.abs(self.y_error)

class ECC(AlignmentMethod):

    def __init__(self, log_file_path, num_iterations=5000, termination_eps=1e-7):
        super().__init__(log_file_path, "cv2ECC", )

        self.num_iterations = num_iterations 
        self.termination_eps = termination_eps

    def run_method(self):

        self.recovered_x_offsets = []
        self.recovered_y_offsets = []
        self.recovered_angles = []

        path_reference = f"{self.dir_path}/{self.out_filenames[0]}"

        for _, file in enumerate(self.out_filenames):

            print(f"Running {self.method_name} on file: {file}")

            path_prealign = f"{self.dir_path}/{file}"

            data_reference = fits.getdata(path_reference, self.hdul_index)
            data_prealign = fits.getdata(path_prealign, self.hdul_index)

            # Convert images to float32
            img1 = np.float32(data_reference)
            img2 = np.float32(data_prealign)

            # convert all NaN values to 0 for ECC to work
            img1 = np.nan_to_num(img1, nan=0.0)
            img2 = np.nan_to_num(img2, nan=0.0)

            # Define the motion model
            warp_mode = cv2.MOTION_EUCLIDEAN  # Translation + Rotation

            # Initialize the matrix to identity
            warp_matrix = np.eye(2, 3, dtype=np.float32)



            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.num_iterations, self.termination_eps)

            # Run the ECC algorithm
            try:
                _, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria)
                h, w = img1.shape[:2]
                
                cx = (w - 1) / 2.0
                cy = (h - 1) / 2.0
                
                R = warp_matrix[:, :2]
                t_matrix = warp_matrix[:, 2]
                
                center = np.array([cx, cy], dtype=np.float32)
                
                translation = t_matrix - center + R @ center
                
                dx = translation[0]
                dy = translation[1]
                
                angle_rad = np.arctan2(R[1, 0], R[0, 0])
                angle_deg = np.degrees(angle_rad)

                print(f"dx={dx:.3f}, dy={dy:.3f}, angle={angle_deg:.3f}")

                print(f"Recovered parameters for {file}: dx={dx}, dy={dy}, angle={angle_deg}")

                self.recovered_x_offsets.append(-dx)
                self.recovered_y_offsets.append(-dy)
                self.recovered_angles.append(-angle_deg)  # Negative because of coordinate system

            except cv2.error as e:
                print(f"ECC failed for {file}: {e}")
                self.recovered_x_offsets.append(0)
                self.recovered_y_offsets.append(0)
                self.recovered_angles.append(0)

        self.angle_error = np.array(self.recovered_angles) - self.rot_angle
        self.angle_error = np.abs(self.angle_error)
        self.angle_error[0] = 0  # first frame is reference frame, so no angle error
        self.x_error = np.array(self.recovered_x_offsets) - np.array(self.x_offsets)
        self.x_error = np.abs(self.x_error)
        self.y_error = np.array(self.recovered_y_offsets) - np.array(self.y_offsets)
        self.y_error = np.abs(self.y_error)
