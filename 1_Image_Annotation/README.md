# TrainYourOwnYOLO: Image Annotation
To train our YOLO detector, we first annotate images located in [`TrainYourOwnYOLO/Data/Source_Images`](/Data/Source_Images) with the help of Microsoft's Visual Object Tagging Tool (VoTT).

## Download VoTT
Head to VoTT [releases](https://github.com/Microsoft/VoTT/releases) and download and install the version for your operation system. 

## Create a New Project

![New Project](/1_Image_Annotation/Screenshots/New_Project.gif)

Create a **New Project** and call it `Annotations`. It is highly recommeded to use `Annotations` as your project name. If you like to use a different name for your project, you will have to modify the command line arguments of subsequent scripts accordingly. 

Under **Source Connection** choose **Add Connection** and put `Images` as **Display Name**. Under **Provider** choose **Local File System** and select the folder with [`Source Images`](/Data/Source_Images). For **Target Connection** choose the same folder as for **Source Connection**. Hit **Save Project** to finish project creation. 

## Export Settings
Navigate to **Export Settings** in the sidebar and then change the **Provider** to `Comma Seperated Values (CSV)`, then hit **Save Export Settings**. 

![New Project](/1_Image_Annotation/Screenshots/Export_Settings.gif)


## Labeling
First create a new tag on the right and give it a relevant tag name. In our example, we choose `Cat_Face`. Then draw bounding boxes around your objects. You can use the number key to quickly assign tags to the current bounding box. 

![New Project](/1_Image_Annotation/Screenshots/Labeling.gif)

## Export Results
Once you have labeled enough images (try to label at least 100 objects) press **CRTL+E** to export the project. You should now see a folder called `vott-csv-export` in the [`Source Images`](/Data/Source_Images) directory. Within that folder, you should see a `*.csv` file called `Annotations-export.csv` which contains information on all bounding boxes. 

## Convert to YOLO Format
As a final step, convert the VoTT csv format to the YOLOv3 format. To do so, run the conversion script:

```
python Convert_to_YOLO_format.py
```
The script generates two output files: `data_train.txt` located in the [`vott-csv-export`](/Data/Source_Images/vott-csv-export)) folder and `data_classes.txt` located in the [`TrainYourOwnYOLO/Data/Model_Weights`](/Data/Model_Weights/) folder.

That's all for annotation! Next, go to [`TrainYourOwnYOLO/2_Training`](/2_Training) to train your YOLOv3 detector.