import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar

FLAGS = None

root_folder = os.path.dirname(os.path.abspath(__file__))
download_folder = os.path.join(root_folder,'src','keras_yolo3')

def downloadIfNeeded(pathWithFileName, url):
    needFile = True; f = None
    try:
        f = open(pathWithFileName)
        needFile = False
    except IOError: pass
    finally:
        if f is not None: f.close()
    if needFile:
        r = requests.get(url,stream=True)
        f = open(pathWithFileName, 'wb')
        file_size = int(r.headers.get('content-length'))
        chunk = 100
        num_bars = file_size // chunk
        bar =  progressbar.ProgressBar(maxval=num_bars).start()
        i = 0
        for chunk in r.iter_content(chunk):
            f.write(chunk)
            bar.update(i)
            i+=1
        f.close()
    else:
        print("Already downloaded, skipping download, of file:", pathWithFileName)

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

    downloadIfNeeded(
        os.path.join(download_folder,'yolov3.weights'),
        'https://pjreddie.com/media/files/yolov3.weights'
    )

    call_string = 'python convert.py yolov3.cfg yolov3.weights yolo.h5'

    subprocess.call(call_string , shell=True, cwd = download_folder )
