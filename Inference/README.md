# EQanalytics: Computer Vision
In this step, we process our previously downloaded training images (`EQanalytics/Data/Street_View_Images`).

## House Segmentation
To segment houses, we run the housing detector script.
```
python detector.py
```
By default, the script runs the detection on images in the folder `EQanalytics/Data/Street_View_Images` and uses the model weights and classes located in `EQanalytics/Data/Model_Weights/Houses`. The segmentation output images are saved to `EQanalytics/Data/House_Detection_Results` and the list of segmentation is also saved in `Housing_Results.csv` in the same folder. Other flags include `--gpu_num`, the number of GPUs to use, `no_save_img`, whether or not to save images with segmentation and  `--postfix`, a string to append to the segmented images.

## House Cropping
Once the houses have been detected, we need to crop out the houses to be passed on to the opening detector. To crop houses, we call the image cropping script.
```
python Crop_Images.py
```
By default, the script takes the images in `EQanalytics/Data/House_Detection_Results` as inputs and saves them under `EQanalytics/Data/House_Cropping_Results` and also to an annotation file called `Cropping_Results.csv` in that same folder. 

## Opening Segmentation

Finally, we run an opening detector by calling  `detector.py` with updated arguments. On ubuntu the call string is as follows.
```
python detector.py --input_images /home/ubuntu/EQanalytics/Data/House_Cropping_Results --classes /home/ubuntu/EQanalytics/Data/Model_Weights/Openings/data_all_classes.txt --output /home/ubuntu/EQanalytics/Data/Opening_Detection_Results --yolo_model /home/ubuntu/EQanalytics/Data/Model_Weights/Openings/trained_weights_final.h5 --box_file /home/ubuntu/EQanalytics/Data/Opening_Detection_Results/Opening_Results.csv --anchors /home/ubuntu/EQanalytics/2_Computer_Vision/src/keras_yolo3/model_data/yolo_anchors.txt
```
To make live easier, one can simply call the opening detector script which is a python wrapper to pass the correct arguments.
```
python opening_detector.py
```
### That's all for Inference!
To train your own object segmentation model, follow the instructions in [Detector_Training](/2_Computer_Vision/Detector_Training/).