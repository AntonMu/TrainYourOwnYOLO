import os
import subprocess
import time
import sys
import argparse
import requests

FLAGS = None

root_folder = os.path.dirname(os.path.abspath(__file__))
download_folder = os.path.join(root_folder,'src','keras_yolo3')

# cd $(dirname $0)
# wget https://pjreddie.com/media/files/yolov3.weights
# python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
if __name__ == '__main__':
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--download_folder", type=str, default=download_folder,
        help = "Folder to download weights to. Default is "+ download_folder
        )

    
    FLAGS = parser.parse_args()

    url = 'https://pjreddie.com/media/files/yolov3.weights'
    r = requests.get(url)

    with open(os.path.join(download_folder,'yolov3.weights'), 'wb') as f:
        f.write(r.content)

    call_string = 'python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5'

    subprocess.call(call_string , shell=True, cwd = download_folder )