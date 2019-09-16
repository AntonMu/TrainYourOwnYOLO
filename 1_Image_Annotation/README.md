# TrainYourOwnYOLO: Image Annotation
To train our YOLO detector, we first annotate images located in [`Data/Source_Images`](TrainYourOwnYOLO/Data/Source_Images) with the help of Microsoft's Visual Object Tagging Tool (VoTT).

## Download VoTT
Head to VoTT [releases](https://github.com/Microsoft/VoTT/releases) and download and install the version for your operation system. 

## Create a New Project

![New Project](/1_Image_Annotation/Screenshots/New_Project.gif)

Create a **New Project** and call it `Annotations`. It is highly recommeded to use `Annotations` as your project name. If you like to use a different name for your project, you will have to modify the command line arguments of subsequent scripts accordingly. 

Under **Source Connection** choose **Add Connection** and put `Images` as **Display Name**. Under **Provider** choose **Local File System** and select the folder with [`Source Images`](TrainYourOwnYOLO/Data/Source_Images). For **Target Connection** choose the same folder as for **Source Connection**. Hit **Save Project** to finish the project creation. 

## Export Settings
Navigate to **Export Settings** in the sidebar and then change the **Provider** to `Comma Seperated Values (CSV)`, then hit **Save Export Settings**. 

![New Project](/1_Image_Annotation/Screenshots/Export_Settings.gif)


## Labeling
Now start labeling process. First create a new tag on the right and give it a relevant tag name. In our example, we choose `Cat_Face`. Then draw bounding boxes around your objects. You can use the number key to quickly assign the tag to the current bounding box. 

![New Project](/1_Image_Annotation/Screenshots/Labeling.gif)


One class called `house` is enough for this task. I recommend to label at least 300 images. The more the better!

![VoTT Houses](/2_Computer_Vision/Detector_Training/Screenshots/VoTT_Houses.png)

 Once you are done, export the project. 
 
![VoTT Saving](/2_Computer_Vision/Detector_Training/Screenshots/VoTT_Save.jpg)

#### Collecting the Result
You should see a folder called `vott-csv-export` which contains all segmented images and a `*.csv` file called `Houses-export.csv`. Copy the contents of this folder to `EQanalytics/Data/vott-csv-export`. 

![VoTT Folder](/2_Computer_Vision/Detector_Training/Screenshots/VoTT_Export.png)

#### Convert to Yolo Format
As a final step, we convert the `VoTT` format to the `YOLOv3` format. To do so, run the `Prepare_Houses_for_Yolo.py` script with:

```
python Prepare_Houses_for_Yolo.py
```

If you have completed all previous steps on your local machine but want to train on AWS, use the flag `--AWS` so the file paths are updated correctly. If you are doing all steps on the same machine, no flags are needed. 

## Starting the Training

Before getting started we need to download the pre-trained YOLOv3 weights. To do so, navigate to `EQanalytics/2_Computer_Vision/src/keras_yolo3`. Next, download the weights and convert them to Keras format:

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
Now, navigate back to `EQanalytics/2_Computer_Vision/Housing_Detector_Training` and run

```
python Train.py
```
Wait until training finishes. Trained weights are saved in `EQanalytics/Data/Model_Weights/Houses` as `trained_weights_final.h5`. That's it - we have successfully trained our housing detector!

# EQanalytics: Opening Detector Training
To train an opening detector, we use the [CMP facade dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). This dataset contains 606 images of building facades and 51769 annotated objects. 

![Annotated Opening Folder](/2_Computer_Vision/Detector_Training/Screenshots/Opening.jpg)

#### Convert to Yolo Format
The first step is to convert the annotations provided in individual `*.xml` files to YOLO format.

```
python Prepare_Openings_for_Yolo.py
```
## Starting the Training
If you haven't already downloaded the pre-trained YOLOv3 weights, follow the instructions under [Starting the Training](#Starting-the-Training). To start the training, run

```
python Train_Opening_detector.py
```

This python script is a wrapper to call the `Train.py` script with the correct arguments. It uses the prepared YOLOv3 training file `EQanalytics/Data/CMP_Facade_DB/data_all_train.txt` for annotations, the file `EQanalytics/Data/Model_Weights/Openings/data_all_classes.txt` for class names and the folder `EQanalytics/Data/Model_Weights/Openings` to save the trained weights to (along with some log files). 

Wait until training finishes. Trained weights are saved in `EQanalytics/Data/Model_Weights/Openings` as `trained_weights_final.h5`. That's it - we have successfully trained our opening detector!