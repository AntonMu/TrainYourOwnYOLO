import os
import subprocess
import time
import sys

root_folder = os.path.dirname(os.path.abspath(__file__))


download_folder = os.path.join(root_folder,'src','keras_yolo3')

convert_script = os.path.join(download_folder,'Download_YOLO_weights.sh')

call_string = 'bash ' + convert_script

subprocess.call(call_string , shell=True)