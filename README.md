# TrainYourOwnYOLO: Building a Custom Image Detector from Scratch

This repo let's you train a custom image detector using the state-of-the-art YOLOv3 computer vision algorithm. For a short write up check out this [medium post](www.medium.com). 

### Pipeline Overview

To build and test your object detection algorithm follow the below steps:

 1. [Image Annotation](/1_Image_Annotation/)
	 - Optional: Download an Image Dataset from Google Images
	 - Install Microsoft's VoTT
	 - Annotate images by Drawing Bounding Boxes
 2. [Training](/2_Training/)
 	- Download pre-trained weights
 	- Train your custom model on annotated images 
 3. [Inference](/3_Inference/)
 	- Detect objects in new images
 
<!-- ### Model Training
The model uses two supervised image detection deep learning approaches (both based on YOLOv3) located in [Detector_Training](/2_Computer_Vision/Detector_Training/).

 1. Train House Identifier
 	- Manually label houses using [VoTT](https://github.com/Microsoft/VoTT).
	- Use transfer learning to train a YOLOv3 detector.
 2. Train Opening Identifier
 	- Use the [CMP facade dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/).
 	- Use transfer learning to train a YOLOv3 detector.

The model also uses un-supervised K-means clustering in the final classification step.  -->
<!-- 
### Inference
Model inference consists of four similar steps. After entering an address (or list of addresses), the corresponding street view images will be downloaded. For all images, the housing model first segments and crops one house per address. Then the opening detector labels all openings and creates a csv file with all dimensions and positions of the openings. Finally, the softness score is determined and used to classify the building as "soft", "non_soft" or "undetermined". 
 -->
## Repo structure
+ [`1_Image_Annotation`](/1_Image_Annotation/): Contains all Information to annotate images
+ [`2_Training`](/2_Training/): Training of your YOLOv3 model
+ [`3_Inference`](/3_Inference/): Testing your trained model on new data
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results

## Getting Started

### Requisites
The code uses python 3.6, Keras with Tensorflow backend. For training it is recommened to us a GPU. For example on an AWS you can use a *p2.xlarge* instance (Tesla K80 GPU with 12GB memory). Inference is generally even fast on a CPU with an average of ~2 images per second. 


### Installation [Linux or Mac]

#### Clone Repo and Install Requirements
Clone this repo with:
```
git clone https://github.com/AntonMu/TrainYourOwnYOLO
cd TrainYourOwnYOLO/
```

Create Virtual Environment ([Venv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) needs to be installed on your system). 
```
python3 -m venv env
source env/bin/activate
```
<!-- 
If you are a conda user, create your virtual environment with:
```
conda create -n EQanalytics
source activate EQanalytics
```
 -->
Next, install all required packages. If you are running TrainYourOwnYOLO on a machine with GPU with CUDA drivers installed run:

```
pip3 install -r requirements.txt
```

Otherwise, run:
```
pip3 install -r requirements_cpu.txt
```

## Quick Start (Inference only)
If you just want to test out the cat face detector on three sample images located in [`Data/Minimal_Example/Minimal_Test_Images`](/Data/Minimal_Example/Minimal_Test_Images) run the `Minimal_Example.py` script with:

```
python Minimal_Example.py
```

The outputs are saved in [`Data/Minimal_Example/Minimal_Test_Images_Results`](/Data/Minimal_Example/Minimal_Test_Images_Results). This includes:
 - Cat pictures with bounding boxes around faces with confidence scores
 - \*.csv file with file names and locations of bounding boxes

## Full Start (Training and Inference)

To also train your model, follow the individual instructions located in [1_Image_Annotation](/1_Image_Annotation/), [2_Training](/2_Training/) and [3_Inference](/3_Inference/), respectively. 

<!-- 
#### Build Environment For Inference

To hit the ground running, download the pre-trained YOLOv3 model weights (235MB) for the housing detector and the pre-trained YOLOv3 weights (236MB) for the opening detector. 
```
bash build/build_inference.sh
```
#### Build Environment For Training
To re-train the housing detector and/or opening detector follow the following steps. 

##### Retrain Housing Detector
To retrain the housing detector, either download your own street view housing image dataset or use the SF vulnerable buildings data set. Once you have created an image folder, install [VoTT](https://github.com/Microsoft/VoTT) on your local machine and segment houses. Alternatively, use the already segmented dataset. Next, download the default YOLOv3 weights to start transfer learning from.  
```
bash build/build_housing_detector.sh --download_images --download_segments
```
The flag, `download_images` indicates that the SF housing image dataset should be downloaded.  The flag, `download_segments` indicates that the manually labeled dataset with houses segmented should be downloaded.

##### Re-train Opening Detector

To retrain the opening detector, either download the [CMP facade dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/) or provide your own data set of segmented building openings (e.g. by using [VoTT](https://github.com/Microsoft/VoTT)) . Also download the default YOLOv3 weights to start transfer learning from.   
```
bash build/build_opening_detector.sh --download_cmp
```
The flag, `download_cmp` indicates that the [CMP facade dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/) should be downloaded.  -->
<!-- 
## Usage
The script doing the work is [logohunter.py](src/logohunter.py) in the `src/` directory. It first uses a custom-trained YOLOv3 to find logo candidates in an image, and then looks for matches between the candidates and a user input logo.

Execute it with the `-h` option to see all of the possible command line inputs. A simple test to match 20 sample input
images in the [data/test/sample_in/](data/test/sample_in/) directory to logos in [data/test/test_brands/](data/test/test_brands/) can be executed with:
```
cd src/
python logohunter.py --test
```
Typical ways to run the program involve specifying one input brand and a folder of sample images:
```
python logohunter.py  --image --input_brands ../data/test/test_brands/test_lexus.png \
                              --input_images ../data/test/lexus/ \
                              --output ../data/test/test_lexus --outtxt

python logohunter.py  --image --input_brands ../data/test/test_brands/test_golden_state.jpg  \
                              --input_images ../data/test/goldenstate/  \
                              --output ../data/test/test_gs --outtxt

python logohunter.py  --image --input_images data_test.txt  \
                              --input_brands ../data/test/test_brands/test_lexus.png  \
                              --outtxt --no_save_img
```

In the first two use cases, we test a folder of images for a single brand ([lexus logo](data/test/test_brands/test_lexus.png) or [golden state logo](data/test/test_brands/test_golden_state.jpg)). The input images were downloaded from Google Images for test purposes. Running LogoHunter saves images with bounding box annotations in the folder specified (`test_lexus`, `test_gs`). Because each of these images contains the logo we are looking for, this is a way to estimate the false negative rate (and the recall).

In the third example, we test a text file containing paths to 2590 input images from the LogosInTheWild dataset against a single brand, without saving the annotated images. Because the brand is new to the dataset, this is a way to estimate the false positive rate (and the precision). (**Note:** this will not run out of the box, as you will need to separately download the LogosInTheWild dataset - follow the instructions below to download the dataset).

#### Data
This project uses the [Logos In The Wild dataset](https://www.iosb.fraunhofer.de/servlet/is/78045/) which can be requested via email directly from the authors of the paper, [arXiv:1710.10891](https://arxiv.org/abs/1710.10891). This dataset includes 11,054 images with 32,850 bounding boxes for a total of 871 brands.

See below for LICENSE information of this dataset.

#### Optional: download, process and clean dataset

Follow the directions in [data/](data/README.md) to download the Logos In The Wild dataset.

#### Optional: train object detection model
After the previous step, the `data_train.txt` and `data_test.txt` files have all the info necessary to train the model. We then follow the instructions of the [keras-yolo3](https://github.com/qqwweee/keras-yolo3) repo: first we download pre-trained YOLO weights from the YOLO official website, and then we convert them to the HDF5 format used by keras.
```
cd src/keras_yolo3
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5

cd ../
python train.py
```
Training detail such as paths to train/text files, log directory, number of epochs, learning rates and so on are specified in `src/train.py`. The training is performed in two runs, first with all the layers except the last three frozen, and then with all layers trainable.

On an AWS EC2 p2.xlarge instance, with a Tesla K-80 GPU with 11GB  of GPU memory and 64GB of RAM, training YOLOv3 for logo detection took approximately 10 hours for 50+50 epochs.
 -->

## License

Unless explicitly stated at the top of a file, all code is licensed under the MIT license.

<!-- 
The Logos In The Wild dataset (links to images, bounding box annotations, clean_dataset.py script) is licensed under the CC-by-SA 4.0 license. The images themselves were crawled from Google Images and are property of their respective copyright owners. For legal reasons, raw images other than the ones in `data/test` are not provided: while this project would fall in the "fair use" category, any commercial application would likely need to generate their own dataset.
 -->
<!-- The model files for the YOLO weights and the extracted logo features are derivative work based off of the Logos In The Wild dataset, and are therefore also licensed under the CC-by-SA 4.0 license. -->

This repo makes use of [**ilmonteux/logohunter**](https://github.com/ilmonteux/logohunter) which itself is inspired by [**qqwweee/keras-yolo3**](https://github.com/qqwweee/keras-yolo3).

## Common Issues

If you are having trouble with getting cv2 to run, try:

```
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
```

If you are using Linux you can are having trouble installing the `*.snap` package file try:
```
snap install - dangerous vott-2.1.0-linux.snap
```
See [Snap Tutorial](https://tutorials.ubuntu.com/tutorial/advanced-snap-usage#2) for more information.