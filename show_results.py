import numpy as np 
import json
import glob
import argparse
import os


class DataContainer:
    """
    A simple container for the data and metadata.
    """

    def __init__(self, metric_file_path):

        try:
            with open(metric_file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON from {metric_file_path}: {e}")

        self.data = data

        self.x_offsets_true = data["x_offsets"]
        self.y_offsets_true = data["y_offsets"]

        
        

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Show results of the pipeline")

    argparser.add_argument(
        "results_dir",
        type=str,
        default="results",
        help="Directory containing the results of the pipeline",
    )

    args = argparser.parse_args()

    results_dir = args.results_dir

    all_files = glob.glob(os.path.join(results_dir, "*_metrics.json"))

    for i, file in enumerate(all_files):

        print(f"Processing file {i + 1}/{len(all_files)}: {file}")

        data_container = DataContainer(file)