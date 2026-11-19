import astroalign as aa
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


REF_PATH = "/home/kostas/Documents/ESO/astroalign-bench/test_data/IMAGE_FOV_prealign_P01_WFI_BB_2018-01-10T02_49_23_0001_alt_001.fits"
SCIENCE_PATH = "/home/kostas/Documents/ESO/astroalign-bench/test_data/IMAGE_FOV_prealign_P01_WFI_BB_2018-01-10T02_49_23_0001_alt_002.fits"


if __name__ == "__main__":
    # Load the input FITS file
    with fits.open(REF_PATH) as ref_hdul:
        ref_data = ref_hdul[1].data

    # Load the science FITS file
    with fits.open(SCIENCE_PATH) as sci_hdul:
        sci_data = sci_hdul[1].data

    sci_data = np.asarray(sci_data)
    ref_data = np.asarray(ref_data)

    sci_data = sci_data.astype(sci_data.dtype.newbyteorder("="), copy=False)
    ref_data = ref_data.astype(ref_data.dtype.newbyteorder("="), copy=False)


    # crop out the borders of the images to avoid edge effects
    ref_data = ref_data[50:-50, 50:-50]
    sci_data = sci_data[50:-50, 50:-50]

    # purge infs and nans:
    ref_data = np.nan_to_num(ref_data, nan=0.0, posinf=0.0, neginf=0.0)
    sci_data = np.nan_to_num(sci_data, nan=0.0, posinf=0.0, neginf=0.0)

    plt.imshow(ref_data, cmap='gray')
    plt.show()

    plt.imshow(sci_data, cmap='gray')
    plt.show()

    # show the diff image:

    diff_image = sci_data - ref_data
    plt.imshow(diff_image, cmap='gray')
    plt.title('Difference Image (Science - Reference)')
    plt.colorbar()
    plt.show()



    transf, (source_list, target_list) = aa.find_transform(sci_data, ref_data)

    # show the sources found:

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(sci_data, cmap='gray')
    ax[0].scatter(source_list[:, 0], source_list[:, 1], s=10, c='red', marker='o', label='Source Points')
    ax[0].set_title('Science Image with Source Points')
    ax[0].legend()

    ax[1].imshow(ref_data, cmap='gray')
    ax[1].scatter(target_list[:, 0], target_list[:, 1], s=10, c='blue', marker='x', label='Target Points')
    ax[1].set_title('Reference Image with Target Points')
    ax[1].legend()

    plt.show()

    registered_image, _ = aa.apply_transform(transf, sci_data, ref_data)

    # plot new diff image:
    new_diff_image = registered_image - ref_data
    plt.imshow(new_diff_image, cmap='gray')
    plt.title('Difference Image (Registered Science - Reference)')
    plt.colorbar()
    plt.show()

    print("Transformation matrix:")
    print(transf)
    print("Inverse transformation matrix:")
    print(transf.inverse)

    x_shape, y_shape = sci_data.shape

    M = transf.params
    R = M[:2, :2]
    t = M[:2, 2]

    center = np.array([(x_shape - 1) / 2, (y_shape - 1) / 2])

    offset = t - center + R @ center

    print("Recovered offset:", offset)

    rotation = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    print("Recovered rotation (degrees):", rotation)




    
