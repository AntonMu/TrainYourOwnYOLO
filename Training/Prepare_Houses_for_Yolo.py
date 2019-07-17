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
sys.path.append(os.path.join(get_parent_dir(1),'Utils'))
from Convert_Format import convert_vott_csv_to_yolo
Data_Folder = os.path.join(get_parent_dir(2),'Data')
VoTT_Folder = os.path.join(Data_Folder,'vott-csv-export')
VoTT_csv =  os.path.join(VoTT_Folder,'Houses-export.csv')
YOLO_filename = os.path.join(VoTT_Folder,'data_train.txt')

model_folder =  os.path.join(Data_Folder,'Model_Weights')
classes_filename = os.path.join(model_folder,'Houses','data_classes.txt')

AWS_path = '/home/ubuntu/EQanalytics/Data/vott-csv-export/'




if __name__ == '__main__':
    # surpress any inhereted default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--VoTT_Folder", type=str, default=VoTT_Folder,
        help = "absolute path to the exported files from the image tagging step with VoTT."
    )

    parser.add_argument(
        "--VoTT_csv", type=str, default=VoTT_csv,
        help = "absolute path to the *.csv file exported from VoTT. The default name is 'Houses-export.csv'."
    )
    parser.add_argument(
        "--YOLO_filename", type=str, default=YOLO_filename,
        help = "absolute path to the file where the annotations in YOLO format should be saved. The default name is 'data_train.txt' and is saved in the VoTT folder."
    )

    parser.add_argument(
        "--item_name", type=str, default='house',
        help = "The name of the annotated item. The default is 'house'."
    )


    parser.add_argument(
        '--AWS', default=False, action="store_true",
        help='Enable this flag if you plan to train on AWS but did your pre-processing on a local machine.'
    )

    FLAGS = parser.parse_args()

    #Prepare the houses dataset for YOLO
    labeldict = dict(zip([FLAGS.item_name],[0,]))
    multi_df = pd.read_csv(FLAGS.VoTT_csv)
    multi_df.drop_duplicates(subset=None, keep='first', inplace=True)
    if FLAGS.AWS:
        train_path = AWS_path
    else:
        train_path = FLAGS.VoTT_Folder
    convert_vott_csv_to_yolo(multi_df,labeldict,path = train_path,target_name=FLAGS.YOLO_filename)
    # Make classes file

    file = open(classes_filename,"w") 
    file.write(FLAGS.item_name) 
    file.close() 


