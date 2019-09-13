import os
import subprocess


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


Data_Folder = os.path.join(get_parent_dir(2),'Data')
Facade_Folder = os.path.join(Data_Folder,'CMP_Facade_DB')
YOLO_filename = os.path.join(Facade_Folder,'data_all_train.txt')
log_dir = os.path.join(Data_Folder,'Model_Weights','Openings')
YOLO_classname = os.path.join(log_dir,'data_all_classes.txt')

def make_call_string(arglist):
	result_string = ''
	for arg in arglist:
		result_string+= ''.join(['--',arg[0],' ', arg[1],' '])
	return result_string

arglist = [['annotation_file',YOLO_filename],['classes_file',YOLO_classname],['log_dir',log_dir]]
call_string = ' '.join(['python Train.py',make_call_string(arglist)])

print('Calling', call_string)
subprocess.call(call_string, shell=True)