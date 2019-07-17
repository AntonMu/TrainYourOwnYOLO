import os
import subprocess
import time
import sys

def make_call_string(arglist):
    result_string = ''
    for arg in arglist:
        result_string+= ''.join(['--',arg[0],' ', arg[1],' '])
    return result_string
root_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root_folder,'Data')
model_folder = os.path.join(data_folder,'Model_Weights')
test_folder =  os.path.join(data_folder,'Minimal_Example')
cropped_folder = os.path.join(test_folder,'Cropped')
street_view_folder = os.path.join(test_folder,'Street_View')
if not os.path.exists(cropped_folder):
    os.mkdir(cropped_folder)
    
#First download the pre-trained weights
download_script = os.path.join(model_folder,'Download_Weights.py')

print('Downloading Pretrained Weights')
start = time.time()
call_string = ' '.join(['python',download_script,'1aPCwYXFAOmklmNMLMh81Yduw5UrbHqkN', os.path.join(model_folder,'Houses','trained_weights_final.h5') ])

subprocess.call(call_string , shell=True)

call_string = ' '.join(['python',download_script, '1FbvHzQWCjucXPbTbI4S1MnBLkAi58Mxv', os.path.join(model_folder,'Openings','trained_weights_final.h5') ])

subprocess.call(call_string , shell=True)
end = time.time()
print('Downloaded Pretrained Weights in {0:.1f} seconds'.format(end-start))

# Now run the housing detector
detector_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),'2_Computer_Vision','detector.py')


houses_result_file =  os.path.join(test_folder, 'Housing_Results.csv')
houses_weights = os.path.join(model_folder,'Houses','trained_weights_final.h5')
houses_classes = os.path.join(model_folder,'Houses','data_classes.txt')
anchors = os.path.join(root_folder,'2_Computer_Vision','src','keras_yolo3','model_data','yolo_anchors.txt')

arglist = [['input_images',street_view_folder],['classes',houses_classes],['output',test_folder],['yolo_model',houses_weights],['box_file',houses_result_file],['anchors',anchors]]
call_string = ' '.join(['python', detector_script,make_call_string(arglist)])

print('Detecting Houses by calling \n\n', call_string,'\n')
start = time.time()
subprocess.call(call_string, shell=True)
end = time.time()
print('Detected Houses in {0:.1f} seconds'.format(end-start))

# #Next, we crop out the houses
cropping_script = os.path.join(root_folder,'2_Computer_Vision','Crop_Images.py')


cropping_result_file = os.path.join(test_folder,'Cropping_Results.csv') 


arglist = [['input_file',houses_result_file],['classes',houses_classes],['output_folder',cropped_folder],['output_file',cropping_result_file]]


call_string = ' '.join(['python', cropping_script,make_call_string(arglist)])

print('Cropping Houses by calling \n\n', call_string,'\n')
start = time.time()
subprocess.call(call_string, shell=True)
end = time.time()
print('Cropped Houses in {0:.1f} seconds'.format(end-start))


# Next run the opening detector

model_folder =  os.path.join(data_folder,'Model_Weights')
opening_weights = os.path.join(model_folder,'Openings','trained_weights_final.h5')
opening_classes = os.path.join(model_folder,'Openings','data_all_classes.txt')
detector_script = os.path.join(root_folder,'2_Computer_Vision','detector.py')
openings_result_file =  os.path.join(test_folder, 'Opening_Results.csv')

arglist = [['input_images',cropped_folder],['classes',opening_classes],['output',test_folder],['yolo_model',opening_weights],['box_file',openings_result_file],['anchors',anchors],['postfix','_opening']]
call_string = ' '.join(['python', detector_script,make_call_string(arglist)])

print('Detecting Openings by calling \n\n', call_string,'\n')
start = time.time()
subprocess.call(call_string, shell=True)
end = time.time()
print('Detected Openings in {0:.1f} seconds'.format(end-start))

#Finally run the classification

classifier_script = os.path.join(root_folder,'3_Classification','Classifier.py')

level_folder =  os.path.join(data_folder,'Level_Detection_Results')
softness_score_file =  os.path.join(test_folder, 'Softness_Scores.csv')

arglist = [['output_file',softness_score_file],['input_file',openings_result_file],['level_folder',test_folder],['classes',opening_classes]]

call_string = ' '.join(['python', classifier_script,make_call_string(arglist)])


print('Calculating Softness Scores by calling \n\n', call_string,'\n')
start = time.time()
subprocess.call(call_string, shell=True)
end = time.time()
print('Calculated Softness Scores in {0:.1f} seconds'.format(end-start))
