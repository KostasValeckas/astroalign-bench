import argparse
from glob import glob
import glob
from methods import AlignmentMethod, MusePipeline, SpacePylot, MinimizeDifference
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

METHOD_ENUM = {
    "MUSE Pipeline": MusePipeline,
    "SpacePylot": SpacePylot,
    "MinimizeDifference": MinimizeDifference,
    "ALL": [MusePipeline, SpacePylot, MinimizeDifference],
}


def evaluate_best_method(method_list):
    """
    Calculate the best method pr. offset based in the errors both in x and y positions.
    """

    all_results = {}

    for method in method_list:
        if (
            method.recovered_x_offsets is not None
            and method.recovered_y_offsets is not None
        ):
            all_results[method.method_name] = {
                "x_error": method.x_error,
                "y_error": method.y_error,
                "cartesian_distance": np.sqrt(
                    np.array(method.x_error) ** 2 + np.array(method.y_error) ** 2
                ),
            }

    best_method = []

    for offset in range(len(method_list[0].x_offsets)):
        best_method_for_offset = None
        best_distance = float("inf")

        for method_name, results in all_results.items():
            distance = results["cartesian_distance"][offset]
            if distance <= best_distance:
                best_distance = distance
                best_method_for_offset = method_name

        best_method.append(best_method_for_offset)

    return best_method


def plot_results(
    method_list: list[AlignmentMethod], output_dir: str, show: bool = False
):
    """
    Parameters
    ----------

    method_list : list[AlignmentMethod]
        List of AlignmentMethod instances to plot results for.
    """

    # filter methods that have recovered offsets
    valid_methods = [
        m
        for m in method_list
        if m.recovered_x_offsets is not None and m.recovered_y_offsets is not None
    ]

    if not valid_methods:
        print("No methods with recovered offsets to plot.")
        return

    n = len(valid_methods)

    # create a GridSpec so the final row (best method) can span both columns
    fig = plt.figure(figsize=(12, 3 * (n + 1)))
    gs = fig.add_gridspec(nrows=n + 2, ncols=3)

    # top row: show original image (use first valid method to find file)
    first = valid_methods[0]
    original_file_path = f"{first.dir_path}/{first.filename}"
    original_hdul = fits.open(original_file_path)
    original_data = original_hdul[first.hdul_index].data

    ax_im = fig.add_subplot(gs[0, 0])
    ax_im.imshow(
        original_data,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=np.nanmedian(original_data) * 2,
    )
    ax_im.set_title("Reference Image")
    ax_im.set_xlabel("X (pixels)")
    ax_im.set_ylabel("Y (pixels)")

    # use the top-right subplot to display diagnostics about the test
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    frames = method_list[0].N_out if method_list else "Unknown"

    blurred_sigma = (
        method_list[0].gaussian_blur_sigma
        if method_list[0].gaussian_blur_sigma != 0
        else False
    )

    info_lines = [
        f"File: {method_list[0].filename}",
        f"N_Frames: {frames}",
        f"Blurred with Gaussian PSF sigma: {blurred_sigma}",
        f"Rotation angle (constant): {method_list[0].rot_angle}",
        f"x offset increment pr. frame: {method_list[0].x_offset}",
        f"y offset increment pr. frame: {method_list[0].y_offset}",
        f"x and y mean erros:",
    ]

    for m in valid_methods:
        try:
            x = np.asarray(m.x_error)
            y = np.asarray(m.y_error)
            if x.size == 0 and y.size == 0:
                info_lines.append(f"{m.method_name}: no recovered offsets")
                continue

            mean_x = np.nanmean(x) if x.size else 0
            mean_y = np.nanmean(y) if y.size else 0

            std_x = np.nanstd(x) if x.size else 0
            std_y = np.nanstd(y) if y.size else 0
            info_lines.append(
                f"{m.method_name}: x [{mean_x:.5f} ± {std_x:.5f}] y [{mean_y:.5f} ± {std_y:.5f}]"
            )
        except Exception:
            info_lines.append(f"{m.method_name}: error reading offsets")

    info_text = "\n".join(info_lines)
    ax_info.text(
        0.01,
        0.99,
        info_text,
        transform=ax_info.transAxes,
        va="top",
        fontsize=10,
        family="monospace",
    )

    original_hdul.close()

    # one row per method
    for idx, method in enumerate(valid_methods):
        row = idx + 1
        ax_x = fig.add_subplot(gs[row, 0])
        ax_y = fig.add_subplot(gs[row, 1])
        ax_angle = fig.add_subplot(gs[row, 2])

        ax_x.plot(method.x_offsets, method.x_error, color="C0")
        ax_y.plot(method.y_offsets, method.y_error, color="C1")
        ax_angle.plot(
            np.arange(len(method.angle_error)), method.angle_error, color="C2"
        )

        ax_x.set_xlabel("Offset in X (pixels)")
        ax_x.set_ylabel("Error in recovered X offset (pixels)")
        ax_x.set_title(f"{method.method_name} — X error")

        ax_y.set_xlabel("Offset in Y (pixels)")
        ax_y.set_ylabel("Error in recovered Y offset (pixels)")
        ax_y.set_title(f"{method.method_name} — Y error")

        ax_angle.set_xlabel("Frame index")
        ax_angle.set_ylabel("Error in recovered rotation angle (degrees)")
        ax_angle.set_title(f"{method.method_name} — Rotation error")

    # create a subplot that spans both columns for the best-method plot
    ax_best_method = fig.add_subplot(gs[n + 1, :])

    best_methods = evaluate_best_method(valid_methods)

    ax_best_method.plot(best_methods, marker="o", linestyle="-", color="C2")
    ax_best_method.set_xlabel("Frame index")
    ax_best_method.set_ylabel("Best Method")
    ax_best_method.set_title("Best Method per Offset")

    plt.tight_layout()
    fig.savefig(
        f"{output_dir}/{method_list[0].filename.split('.')[0]}_results.pdf",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_results_to_file(method_list: list[AlignmentMethod], output_dir: str):
    """
    Write the results of the alignment methods to a JSON file.

    Parameters
    ----------
    method_list : list[AlignmentMethod]
        List of AlignmentMethod instances to write results for.
    output_dir : str
        Directory where the results file will be saved.
    """

    results = {}

    results["x_offsets"] = (method_list[0].x_offsets.tolist(),)
    results["y_offsets"] = (method_list[0].y_offsets.tolist(),)
    results["best_method"] = evaluate_best_method(method_list)
    results["gaussian_blur_sigma"] = method_list[0].gaussian_blur_sigma
    results["rotation_angle"] = method_list[0].rot_angle

    for method in method_list:

        results[method.method_name] = {
            "recovered_x_offsets": method.recovered_x_offsets,
            "recovered_y_offsets": method.recovered_y_offsets,
            "x_error": (
                method.x_error.tolist() if method.x_error is not None else None
            ),
            "y_error": (
                method.y_error.tolist() if method.y_error is not None else None
            ),
            "recovered_angles": (
                method.recovered_angles if method.recovered_angles is not None else None
            ),
            "angle_error": (
                method.angle_error.tolist() if method.angle_error is not None else None
            ),
        }

    output_file_path = (
        f"{output_dir}/{method_list[0].filename.split('.')[0]}_metrics.json"
    )

    with open(output_file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)

    print(f"Results written to {output_file_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run allignment tests.")

    parser.add_argument(
        "log_file_path",
        nargs="?",
        default=".",
        help="Path to the log file containing test parameters. "
        "If --all_log_files is set, this should be a directory containing multiple log files."
        "Else, it should be a single JSON log file.",
    )

    parser.add_argument(
        "--output_dir",
        nargs="?",
        default=".",
        help="Path to the output directory where results will be saved (default: current directory)",
    )

    parser.add_argument(
        "--all_log_files",
        action="store_true",
        help="If set, run all JSON log files in the provided log_file_path directory",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="ALL",
        help="Alignment method to use (default: ALL)."
        "Options: 'MUSE Pipeline', 'SpacePylot', or 'ALL' to run all methods.",
    )

    parser.add_argument(
        "--delete_files",
        action="store_true",
        help="If ",
    )

    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="If set, display the plots generated by the tests.",
    )

    args = parser.parse_args()

    log_file_path = args.log_file_path
    all_log_files = args.all_log_files

    directory_path = args.output_dir

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    method_name = args.method
    method = METHOD_ENUM.get(method_name)

    if all_log_files:

        log_files = glob.glob(f"{log_file_path}/*.json")

        # quick sanity check to reject garbage
        valid = []

        for i, file in enumerate(log_files):
            try:
                log_dir = json.load(open(file, "r"))

            except FileNotFoundError:
                print(f"Log file not found: {file}")
                raise

            except json.JSONDecodeError:
                print(f"Invalid JSON in log file: {file}")
                raise

            if log_dir.get("filename") is None or log_dir.get("dir_path") is None:
                # remove the file from the list if it doesn't have the required keys
                print(f"Log file {file} does not contain required keys. Skipping.")
                continue

            else:
                valid.append(file)

        log_files = valid.copy()
    else:

        # a bit hacky but less code
        log_files = [log_file_path]

    for log_file in log_files:

        method_list = []

        print(f"Running tests for log file: {log_file}")

        if method_name == "ALL":

            for method_class in METHOD_ENUM["ALL"]:

                method_instance = method_class(log_file)

                method_instance.run_method()

                method_list.append(method_instance)

        else:

            if method is None:

                print(f"Unknown method: {method_name}")

                continue

            method_instance = method(log_file)

            method_instance.run_method()

            method_list.append(method_instance)

        plot_results(method_list, directory_path, show=args.show_plots)
        write_results_to_file(method_list, directory_path)

        if args.delete_files:
            for file in method_list[0].out_filenames:
                file_path = f"{method_list[0].dir_path}/{file}"
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted {file_path}")
                else:
                    print(f"File {file_path} does not exist, cannot delete.")
