import os
import subprocess
import time
import sys

# TrainYourOwnYOLO/Training/src/keras_yolo3

root_folder = os.path.dirname(os.path.abspath(__file__))


download_folder = os.path.join(root_folder,'src','keras_yolo3')

convert_script = os.path.join(download_folder,'Download_YOLO_weights.sh')

call_string = 'bash ' + convert_script

subprocess.call(call_string , shell=True)


# weight_file = os.path.join(download_folder,'yolov3.weights')
# print(download_folder)
# # wget https://pjreddie.com/media/files/yolov3.weights
# # python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

# # call_string = 'wget https://pjreddie.com/media/files/yolov3.weights -O ' + weight_file

# # subprocess.call(call_string , shell=True)


# # wget http://download.files.com/software/files.tar.gz -O /home/yourname/Downloads

# convert_script = weight_file = os.path.join(download_folder,'convert.py')

# call_string = 'python ' + convert_script + ' yolov3.cfg yolov3.weights model_data/yolo.h5'
# subprocess.call(call_string , shell=True)

# # python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5


# def make_call_string(arglist):
#     result_string = ''
#     for arg in arglist:
#         result_string+= ''.join(['--',arg[0],' ', arg[1],' '])
#     return result_string
# root_folder = os.path.dirname(os.path.abspath(__file__))
# data_folder = os.path.join(root_folder,'Data')
# model_folder = os.path.join(data_folder,'Model_Weights')
# image_folder = os.path.join(data_folder,'Source_Images')
# input_folder =  os.path.join(image_folder,'Test_Images')
# output_folder =  os.path.join(image_folder,'Test_Image_Detection_Results')


# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)
    
# #First download the pre-trained weights
# download_script = os.path.join(model_folder,'Download_Weights.py')

# print('Downloading Pretrained Weights')
# start = time.time()
# call_string = ' '.join(['python',download_script,'1MGXAP_XD_w4OExPP10UHsejWrMww8Tu7', os.path.join(model_folder,'trained_weights_final.h5') ])

# subprocess.call(call_string , shell=True)

# end = time.time()
# print('Downloaded Pretrained Weights in {0:.1f} seconds'.format(end-start))

# # Now run the cat face detector
# detector_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),'3_Inference','Detector.py')


# result_file =  os.path.join(output_folder, 'Cat_Faces_Results.csv')
# model_weights = os.path.join(model_folder,'trained_weights_final.h5')
# classes_file = os.path.join(model_folder,'data_classes.txt')
# anchors = os.path.join(root_folder,'2_Training','src','keras_yolo3','model_data','yolo_anchors.txt')

# arglist = [['input_images',input_folder],['classes',classes_file],['output',output_folder],['yolo_model',model_weights],['box_file',result_file],['anchors',anchors]]
# call_string = ' '.join(['python', detector_script,make_call_string(arglist)])

# print('Detecting Cat Faces by calling \n\n', call_string,'\n')
# start = time.time()
# subprocess.call(call_string, shell=True)
# end = time.time()
# print('Detected Cat Faces in {0:.1f} seconds'.format(end-start))
