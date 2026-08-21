import numpy as np
import json
import glob
import argparse
import os
import matplotlib.pyplot as plt


# These are for plotting:

LARGE_X_OFFSET = 0.5
LARGE_Y_OFFSET = 0.5

SMALL_X_OFFSET = 0.02
SMALL_Y_OFFSET = 0.02

N_FRAMES = 80

SMALL_ANGLE = 0.1
LARGE_ANGLE = 1

LARGE_X_OFFSET_INTERVAL = [LARGE_X_OFFSET, LARGE_X_OFFSET * N_FRAMES]
LARGE_Y_OFFSET_INTERVAL = [LARGE_Y_OFFSET, LARGE_Y_OFFSET * N_FRAMES]

SMALL_X_OFFSET_INTERVAL = [SMALL_X_OFFSET, SMALL_X_OFFSET * N_FRAMES]
SMALL_Y_OFFSET_INTERVAL = [SMALL_Y_OFFSET, SMALL_Y_OFFSET * N_FRAMES]

BLURRING_FWHM = 2


# These are the hardcoded sub-directory names that are used to construct
# the list of ouput sub-directories for a test run.
results_dir_list = [
    "_results_blurr_rot_shift",
    "_results_blurr_rot",
    "_results_blurr_small_rot_shift",
    "_results_blurr_small_rot",
    "_results_blurr_shift",
    "_results_blurr_sub_rot_shift",
    "_results_blurr_sub_small_rot_shift",
    "_results_blurr_sub_shift",
    "_results_rot_shift",
    "_results_small_rot_shift",
    "_results_shift",
    "_results_rot",
    "_results_small_rot",
    "_results_blurr",
    "_results_sub_rot_shift",
    "_results_sub_small_rot_shift",
    "_results_sub_shift",
    "_results_control",
]

# This is a mapping of different test parameters for easier representation

mapping = {
    "_results_blurr_rot_shift": [
        ("Blurred", True),
        ("Large_rot", True),
        ("Large_shift", True),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_blurr_rot": [
        ("Blurred", True),
        ("Large_rot", True),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_blurr_small_rot_shift": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", True),
        ("Small_rot", True),
        ("Small_shift", False),
    ],
    "_results_blurr_small_rot": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", True),
        ("Small_shift", False),
    ],
    "_results_blurr_shift": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", True),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_blurr_sub_rot_shift": [
        ("Blurred", True),
        ("Large_rot", True),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", True),
    ],
    "_results_blurr_sub_small_rot_shift": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", True),
        ("Small_shift", True),
    ],
    "_results_blurr_sub_shift": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", True),
    ],
    "_results_rot_shift": [
        ("Blurred", False),
        ("Large_rot", True),
        ("Large_shift", True),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_small_rot_shift": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", True),
        ("Small_rot", True),
        ("Small_shift", False),
    ],
    "_results_shift": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", True),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_rot": [
        ("Blurred", False),
        ("Large_rot", True),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_small_rot": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", True),
        ("Small_shift", False),
    ],
    "_results_blurr": [
        ("Blurred", True),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
    "_results_sub_rot_shift": [
        ("Blurred", False),
        ("Large_rot", True),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", True),
    ],
    "_results_sub_small_rot_shift": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", True),
        ("Small_shift", True),
    ],
    "_results_sub_shift": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", True),
    ],
    "_results_control": [
        ("Blurred", False),
        ("Large_rot", False),
        ("Large_shift", False),
        ("Small_rot", False),
        ("Small_shift", False),
    ],
}


# these are for sorting out entries in the metrics files
skip_keys = [
    "x_offsets",
    "y_offsets",
    "best_method",
    "gaussian_blur_sigma",
    "rotation_angle",
]


def show_mapping():
    """
    Visual representration of the test matrix mapping.
    """
    # Convert mapping into a matrix
    columns = [
        "Blurred PSF",
        "Large Rotation",
        "Large Shift",
        "Small Rotation",
        "Small Shift",
    ]

    rows = list(mapping.keys())

    data = np.array([[value for _, value in mapping[row]] for row in rows], dtype=int)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # True = green, False = red
    ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Column labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=11)

    # Row labels
    # Remove "_results_" to make the labels cleaner
    clean_rows = [row.replace("_results_", "") for row in rows]

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(clean_rows, fontsize=10)

    # Add ✓ / ✗ inside each cell
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                "✓" if data[i, j] else "✗",
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    # Add grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)

    # Remove minor tick marks
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Testing Setup", fontsize=15, pad=15)
    ax.set_xlabel("Test condition", fontsize=12)
    ax.set_ylabel("Results case", fontsize=12)

    plt.tight_layout(h_pad=2.0)
    plt.show()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Show results of the pipeline")

    argparser.add_argument(
        "results_dir",
        type=str,
        default="results",
        help="Directory containing the results of the pipeline",
    )

    argparser.add_argument(
        "prefix",
        type=str,
        default="",
        help="Prefix of the results directories",
    )

    args = argparser.parse_args()

    show_mapping()

    results = {}

    
    for directory_suffix in results_dir_list:

        results[directory_suffix] = {}

        output_path = os.path.join(args.results_dir, args.prefix + directory_suffix)

        test_setup = mapping[directory_suffix]

        all_metrics_files = glob.glob(os.path.join(output_path, "*_metrics.json"))

        for file in all_metrics_files:

            results_dict = json.load(open(file, "r"))

            keys = list(results_dict.keys())

            result_temp = {}

            # prepare temp for processing
            for key in keys:

                if key in skip_keys:
                    continue

                if key not in results[directory_suffix]:
                    results[directory_suffix][key] = {
                        "x_mean_errors": [],
                        "x_std_dev": [],
                        "y_mean_errors": [],
                        "y_std_dev": [],
                        "angle_mean_errors": [],
                        "angle_std_dev": [],
                    }

                result_temp[key] = results_dict[key]

            for key, dict in result_temp.items():

                # we skip the first entry as it is the reference frame and has no error
                x_errors = dict["x_error"][1:]
                print(x_errors)
                y_errors = dict["y_error"][1:]
                angle_error = dict["angle_error"][1:]

                x_mean_error = np.mean(x_errors)
                x_std_dev = np.std(x_errors)

                y_mean_error = np.mean(y_errors)
                y_std_dev = np.std(y_errors)

                angle_mean_error = np.mean(angle_error)
                angle_std_dev = np.std(angle_error)

                results[directory_suffix][key]["x_mean_errors"].append(x_mean_error)
                results[directory_suffix][key]["x_std_dev"].append(x_std_dev)

                results[directory_suffix][key]["y_mean_errors"].append(y_mean_error)
                results[directory_suffix][key]["y_std_dev"].append(y_std_dev)

                results[directory_suffix][key]["angle_mean_errors"].append(
                    angle_mean_error
                )
                results[directory_suffix][key]["angle_std_dev"].append(angle_std_dev)

    # Show mapping and error heatmaps in a 2x2 grid.
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))

    # Top-left: same output as show_mapping()
    ax = axs[0, 0]
    columns = [
        f"Blurred PSF ({BLURRING_FWHM} fwhm Gaussian kernel)",
        f"Large Rotation ({LARGE_ANGLE} deg)",
        f"Large Shift {LARGE_X_OFFSET_INTERVAL} x, {LARGE_Y_OFFSET_INTERVAL} y",
        f"Small Rotation ({SMALL_ANGLE} deg)",
        f"Small Shift {SMALL_X_OFFSET_INTERVAL} x, {SMALL_Y_OFFSET_INTERVAL} y",
    ]
    rows = list(mapping.keys())
    mapping_data = np.array(
        [[value for _, value in mapping[row]] for row in rows], dtype=int
    )
    ax.imshow(mapping_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=6)
    clean_rows = [row.replace("_results_", "") for row in rows]
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(clean_rows, fontsize=10)
    """
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                "✓" if mapping_data[i, j] else "✗",
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )
    """
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(f"Testing Setup ({N_FRAMES * len(all_metrics_files)} files pr. setup)", fontsize=12, pad=5)
    # ax.set_xlabel("Test condition", fontsize=12)
    # ax.set_ylabel("Results case", fontsize=12)

    method_names = list(results[results_dir_list[0]].keys())
    clean_rows = [row.replace("_results_", "") for row in results_dir_list]

    plot_specs = [
        (0, 1, "angle", "Mean Angle Errors "),
        (1, 0, "x", "Mean X Errors "),
        (1, 1, "y", "Mean Y Errors "),
    ]

    for row_idx, col_idx, error_type, title in plot_specs:
        ax = axs[row_idx, col_idx]

        data = np.array(
            [
                [
                    np.mean(results[directory_suffix][key][f"{error_type}_mean_errors"])
                    for key in results[directory_suffix]
                ]
                for directory_suffix in results_dir_list
            ]
        )

        std_data = np.array(
            [
                [
                    np.mean(results[directory_suffix][key][f"{error_type}_std_dev"])
                    for key in results[directory_suffix]
                ]
                for directory_suffix in results_dir_list
            ]
        )

        cax = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.5g}± {std_data[i, j]:.5g}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=6,
                )

        ax.set_xticks(np.arange(len(method_names)))
        ax.set_xticklabels(method_names, fontsize=8, rotation=5, ha="right")
        ax.set_yticks(np.arange(len(clean_rows)))
        ax.set_yticklabels(clean_rows, fontsize=8)

        cbar = fig.colorbar(cax, ax=ax)
        # cbar.set_label(f'Mean {error_type.upper()} Error', rotation=270, labelpad=15)

        ax.set_title(title, fontsize=10, pad=5)
        # ax.set_xlabel("Method", fontsize=12)
        # ax.set_ylabel("Test Case", fontsize=12)

    plt.tight_layout(h_pad=2.0)
    plt.show()
