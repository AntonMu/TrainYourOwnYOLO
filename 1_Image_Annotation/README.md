# TrainYourOwnYOLO: Image Annotation

## Dataset
To train the YOLO object detector on your own dataset, copy your training images to [`TrainYourOwnYOLO/Data/Source_Images/Training_Images`](/Data/Source_Images/Training_Images/). By default, this directory is pre-populated with 101 cat images. Feel free to delete all existing cat images to make your project cleaner. 

### Creating a Dataset from Scratch
If you do not already have an image dataset, consider using a Chrome extension such as [Fatkun Batch Downloader](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en) which lets you search and download images from Google Images. For instance, you can build a fidget spinner detector by searching for images with fidget spinners. 

## Annotation
To make our detector learn, we first need to feed it some good training examples. We use Microsoft's Visual Object Tagging Tool (VoTT) to manually label images in our training folder [`TrainYourOwnYOLO/Data/Source_Images/Training_Images`](/Data/Source_Images/Training_Images/). To achieve decent results annotate at least 100 images. For good results label at least 300 images and for great results label 1000+ images. 

### Download VoTT
Head to VoTT [releases](https://github.com/Microsoft/VoTT/releases) and download and install the version for your operating system. Under `Assets` select the package for your operating system: 
- `vott-2.x.x-darwin.dmg` for Mac users, 
- `vott-2.x.x-win32.exe` for Windows users and 
- `vott-2.x.x-linux.snap` for Linux users.

Installing `*.snap` files requires the snapd package manager which is available at [snapcraft.io](https://snapcraft.io/docs/installing-snapd).

### Create a New Project

Create a **New Project** and call it `Annotations`. It is highly recommended to use `Annotations` as your project name. If you like to use a different name for your project, you will have to modify the command line arguments of subsequent scripts accordingly. 

Under **Source Connection** choose **Add Connection** and put `Images` as **Display Name**. Under **Provider** choose **Local File System** and select [`TrainYourOwnYOLO/Data/Source Images/Training_Images`](/Data/Source_Images/Training_Images) and then **Save Connection**. For **Target Connection** choose the same folder as for **Source Connection**. Hit **Save Project** to finish project creation. 

![New Project](/1_Image_Annotation/Screen_Recordings/New_Project.gif)

### Export Settings
Navigate to **Export Settings** in the sidebar and then change the **Provider** to `Comma Separated Values (CSV)`, then hit **Save Export Settings**. 

![New Project](/1_Image_Annotation/Screen_Recordings/Export_Settings.gif)


### Labeling
First create a new tag on the right and give it a relevant tag name. In our example, we choose `Cat_Face`. Then draw bounding boxes around your objects. You can use the number key **1** to quickly assign the first tag to the current bounding box. 

![New Project](/1_Image_Annotation/Screen_Recordings/Labeling.gif)

### Export Results
Once you have labeled enough images press **CRTL+E** to export the project. You should now see a folder called [`vott-csv-export`](/Data/Source_Images/Training_Images/vott-csv-export) in the [`Training_Images`](/Data/Source_Images/Training_Images) directory. Within that folder, you should see a `*.csv` file called [`Annotations-export.csv`](/Data/Source_Images/Training_Images/vott-csv-export/Annotations-export.csv) which contains file names and bounding box coordinates. 

## Convert to YOLO Format
As a final step, convert the VoTT csv format to the YOLOv3 format. To do so, run the conversion script from within the [`TrainYourOwnYOLO/1_Image_Annotation`](/1_Image_Annotation/) folder:

```
python Convert_to_YOLO_format.py
```
The script generates two output files: [`data_train.txt`](/Data/Source_Images/Training_Images/vott-csv-export/data_train.txt) located in the [`TrainYourOwnYOLO/Data/Source_Images/Training_Images/vott-csv-export`](/Data/Source_Images/Training_Images/vott-csv-export) folder and [`data_classes.txt`](/Data/Model_Weights/data_classes.txt) located in the [`TrainYourOwnYOLO/Data/Model_Weights`](/Data/Model_Weights/) folder. To list available command line options run `python Convert_to_YOLO_format.py -h`.

### That's all for annotation! 
Next, go to [`TrainYourOwnYOLO/2_Training`](/2_Training) to train your YOLOv3 detector.