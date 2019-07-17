from PIL import Image
from os import path, makedirs
import os
import re 
import pandas as pd
import sys
import argparse

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Utils')
sys.path.append(utils_path)

from Get_File_Paths import GetFileList
from Convert_Format import crop_and_save

data_folder = os.path.join(get_parent_dir(n=1),'Data')

houses_file =  os.path.join(data_folder,'House_Detection_Results', 'Housing_Results.csv')

cropping_result_folder = os.path.join(data_folder,'House_Cropping_Results') 
cropping_result_file = os.path.join(data_folder,'House_Cropping_Results','Cropping_Results.csv') 

houses_classes = os.path.join(data_folder,'Model_Weights','Houses','data_classes.txt')


FLAGS = None


if __name__ == '__main__':
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''

    parser.add_argument(
        "--input_file", type=str, default=houses_file,
        help = "Path to *.csv with YOLO detection results."
    )

    parser.add_argument(
        "--output_folder", type=str, default=cropping_result_folder,
        help = "Output folder for cropped images."
    )

    parser.add_argument(
        "--output_file", type=str, default=cropping_result_file,
        help = "Output file with list of file paths for all cropped images."
    )
    
    parser.add_argument(
        '--postfix', type=str, dest = 'postfix', default = 'cropped',
        help='Specify the postfix to attach to cropped images'
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path', default = houses_classes,
        help='path to YOLO class specifications'
    )

    parser.add_argument(
        '--one', type=bool, default = True,
        help='If True, then for each input image only the most central object will be cropped and saved.'
    )
    

    FLAGS = parser.parse_args()

    class_file = open(FLAGS.classes_path, 'r')
    input_labels = [line.rstrip('\n') for line in class_file.readlines()]
    label_dict = dict(zip(list(range(len(input_labels))),input_labels))


    image_df = pd.read_csv(FLAGS.input_file)
    crop_and_save(image_df,target_path = FLAGS.output_folder, target_file= FLAGS.output_file, label_dict = label_dict, postfix=FLAGS.postfix, one = FLAGS.one)       