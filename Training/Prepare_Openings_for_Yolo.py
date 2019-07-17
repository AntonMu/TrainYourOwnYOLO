from PIL import Image
from os import path, makedirs
import os
import re 
import pandas as pd
import sys
import argparse
import numpy as np
def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path
sys.path.append(os.path.join(get_parent_dir(1),'Utils'))
from Convert_Format import convert_vott_csv_to_yolo, csv_from_xml
from Get_File_Paths import ChangeToOtherMachine

Data_Folder = os.path.join(get_parent_dir(2),'Data')
CMP_Folder = os.path.join(Data_Folder,'CMP_Facade_DB')
CSV_filename = os.path.join(CMP_Folder,'Annotations.csv')

model_folder =  os.path.join(Data_Folder,'Model_Weights')
classes_filename = os.path.join(model_folder,'Openings','data_all_classes.txt')

YOLO_filename = os.path.join(CMP_Folder,'data_all_train.txt')
AWS_path = '/home/ubuntu/'

if __name__ == '__main__':
    # surpress any inhereted default values
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        "--YOLO_filename", type=str, default=YOLO_filename,
        help = "absolute path to the file where the annotations in YOLO format should be saved. The default name is 'data_train.txt' and is saved in the VoTT folder."
    )

    parser.add_argument(
        "--drop_classes", default=True, action="store_false",
        help = "If enabled, only relevant classes will be trained on. That is, only 'shop', 'window' and 'door'."
    )

    parser.add_argument(
        '--single_class', default=False, action="store_true",
        help='Enable this flag if you want to only train on one class called "openings".'
    )


    parser.add_argument(
        '--AWS', default=True, action="store_false",
        help='Enable this flag if you plan to train on AWS but did your pre-processing on a local machine.'
    )


    FLAGS = parser.parse_args()

    df_csv = csv_from_xml(CMP_Folder)
    # Make sure the min label code is 0 
    df_csv['code'] = df_csv['code'].astype(int)-min(df_csv['code'].astype(int).values)


    if FLAGS.AWS:
        df_csv['image_path']=ChangeToOtherMachine(df_csv['image_path'].values,remote_machine=AWS_path)
    df_csv.to_csv(CSV_filename,index=False)

    #Get label names and sort 
    if FLAGS.drop_classes:
    	df_csv = df_csv[df_csv['label'].isin(['door','window','shop'])]
    	codes = df_csv['code'].unique()
    	df_csv['code']=df_csv['code'].apply(lambda x: np.where(codes==x)[0][0])
    if FLAGS.single_class:
    	df_csv['code']=0
    	df_csv['label']='opening'
    df_csv.to_csv('est.csv')
    sorted_names = ((df_csv.drop_duplicates(subset = ['code','label'])[['code','label']].sort_values(by = ['code']))['label']).values

    #Write sorted names to file to make classes file
    with open(classes_filename, 'w') as f:
        for name in sorted_names[:-1]:
            f.write("%s\n" % name)
        f.write("%s" % sorted_names[-1])
    # Convert Vott csv format to YOLO format
    convert_vott_csv_to_yolo(df_csv,abs_path = True,target_name=FLAGS.YOLO_filename)


