import subprocess
import os

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(root_folder,'Data')
model_folder = os.path.join(data_folder,'Model_Weights')
openings_input_folder = os.path.join(data_folder,'House_Cropping_Results')
openings_weights = os.path.join(model_folder,'Openings','trained_weights_final.h5')
openings_classes = os.path.join(model_folder,'Openings','data_all_classes.txt')
openings_result_folder = os.path.join(data_folder,'Opening_Detection_Results') 
openings_result_file =  os.path.join(openings_result_folder, 'Opening_Results.csv')
anchors = os.path.join(root_folder,'2_Computer_Vision','src','keras_yolo3','model_data','yolo_anchors.txt')

postfix = '_opening'

def make_call_string(arglist):
    result_string = ''
    for arg in arglist:
        result_string+= ''.join(['--',arg[0],' ', arg[1],' '])
    return result_string

arglist = [['input_images',openings_input_folder],['classes',openings_classes],['output',openings_result_folder],['yolo_model',openings_weights],['box_file',openings_result_file],['anchors',anchors]]
call_string = ' '.join(['python detector.py',make_call_string(arglist)])

print('Running Opening Detector by calling \n\n', call_string,'\n')
subprocess.call(call_string, shell=True)