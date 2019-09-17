# TrainYourOwnYOLO: Building a Custom Image Detector from Scratch [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This repo let's you train a custom image detector using the state-of-the-art [YOLOv3](https://pjreddie.com/darknet/yolo/) computer vision algorithm. For a short write up check out this [medium post](https://medium.com/@muehle/how-to-train-your-own-yolov3-detector-from-scratch-224d10e55de2). 

### Pipeline Overview

To build and test your object detection algorithm follow the below steps:

 1. [Image Annotation](/1_Image_Annotation/)
	 - Install Microsoft's Visual Object Tagging Tool (VoTT)
	 - Annotate images
 2. [Training](/2_Training/)
 	- Download pre-trained weights
 	- Train your custom YOLO model on annotated images 
 3. [Inference](/3_Inference/)
 	- Detect objects in new images

## Repo structure
+ [`1_Image_Annotation`](/1_Image_Annotation/): Scripts and instructions on annotating images
+ [`2_Training`](/2_Training/): Scripts and instructions on training your YOLOv3 model
+ [`3_Inference`](/3_Inference/): Scripts and instructions on testing your trained YOLO model on new images
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
+ [`Utils`](/Utils/): Contains utility scripts used by main scripts

## Getting Started

### Requisites
The code uses python 3.6, Keras with Tensorflow backend. For training it is recommended to use a GPU. For example on an AWS you can use a **p2.xlarge** instance (Tesla K80 GPU with 12GB memory). Inference is very fast even on a CPU with an average of ~2 images per second. 


### Installation

#### Setting up Environment [Linux or Mac]
<!-- #### Clone Repo and Install Requirements -->
Clone this repo with:
```
git clone https://github.com/AntonMu/TrainYourOwnYOLO
cd TrainYourOwnYOLO/
```
Create Virtual Environment (requires [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) which is included in the standard library of Python 3.3 or newer):
```
python3 -m venv env
source env/bin/activate
```
Make sure that, from now on, you run all commands from within your virtual environment.

#### Setting up Environment [Windows]
Use the [Github Desktop GUI](https://desktop.github.com/) to clone this repo to your local machine. Navigate to the `TrainYourOwnYOLO` project folder and open a power shell window by pressing **Shift + Right Click** and selecting `Open PowerShell window here` in the drop-down menu.

Create Virtual Environment (requires [venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) which is included in the standard library of Python 3.3 or newer):

```
py -m venv env
.\env\Scripts\activate
```
![PowerShell](/Utils/Screenshots/PowerShell.png)
Make sure that, from now on, you run all commands from within your virtual environment.

#### Install Required Packages [Windows, Mac or Linux]
If you are running TrainYourOwnYOLO on a machine with GPU with CUDA drivers installed run:

```
pip3 install -r requirements_gpu.txt
```
Otherwise, run:
```
pip3 install -r requirements_cpu.txt
```

## Quick Start (Inference only)
To test the cat face detector on a few test images located in [`Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images) run the `Minimal_Example.py` script in the root folder with:

```
python Minimal_Example.py
```

The outputs are saved in [`Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). This includes:
 - Cat pictures with bounding boxes around faces with confidence scores and
 - [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) file with file names and locations of bounding boxes.

## Full Start (Training and Inference)

To train your own model, follow the individual instructions located in [`1_Image_Annotation`](/1_Image_Annotation/), [`2_Training`](/2_Training/) and [`3_Inference`](/3_Inference/), respectively. 

## License

Unless explicitly stated otherwise at the top of a file, all code is licensed under the MIT license. This repo makes use of [**ilmonteux/logohunter**](https://github.com/ilmonteux/logohunter) which itself is inspired by [**qqwweee/keras-yolo3**](https://github.com/qqwweee/keras-yolo3).

## Common Issues

If you are having trouble with getting cv2 to run, try:

```
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
```

If you are using Linux and having trouble installing `*.snap` package files try:
```
snap install - dangerous vott-2.1.0-linux.snap
```
See [Snap Tutorial](https://tutorials.ubuntu.com/tutorial/advanced-snap-usage#2) for more information.