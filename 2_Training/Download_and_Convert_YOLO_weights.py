import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar

FLAGS = None

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
download_folder = os.path.join(root_folder, "2_Training", "src", "keras_yolo3")
data_folder = os.path.join(root_folder, "Data")
model_folder = os.path.join(data_folder, "Model_Weights")
download_script = os.path.join(model_folder, "Download_Weights.py")

if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """
    parser.add_argument(
        "--download_folder",
        type=str,
        default=download_folder,
        help="Folder to download weights to. Default is " + download_folder,
    )

    FLAGS = parser.parse_args()

    if not os.path.isfile(os.path.join(download_folder, "yolov3.weights")):
        print("\n", "Downloading Raw YOLOv3 Weights", "\n")
        start = time.time()
        call_string = " ".join(
            [
                "python",
                download_script,
                "1ENKguLZbkgvM8unU3Hq1BoFzoLeGWvE_",
                os.path.join(download_folder, "yolov3.weights"),
            ]
        )

        subprocess.call(call_string, shell=True)

        end = time.time()
        print(
            "Downloaded Raw YOLOv3 Weights in {0:.1f} seconds".format(end - start), "\n"
        )

        # Original URL: https://pjreddie.com/media/files/yolov3.weights

        call_string = "python convert.py yolov3.cfg yolov3.weights yolo.h5"

        subprocess.call(call_string, shell=True, cwd=download_folder)
