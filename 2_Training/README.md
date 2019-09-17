# TrainYourOwnYOLO: Training
Using the training images located in [`TrainYourOwnYOLO/Data/Source_Images`](/Data/Source_Images) and the annotation file [`data_train.txt`](/Data/Source_Images/vott-csv-export) which we have created in the [previous step](/1_Image_Annotation/) we are now ready to train our YOLOv3 detector. 

## Download and Convert Pre-Trained Weights
Before getting started download the pre-trained YOLOv3 weights and convert them to the keras format. To run both steps execute:

```
python Download_and_Convert_YOLO_weights.py
```
## Train YOLOv3 Detector
To start the training, run the training script:
```
python Train_YOLO.py 
```
Depending on your set-up, this process can take a few minutes to a few hours. To speed up training consider using a GPU. The final weights are saved in [`TrainYourOwnYOLO/Data/Model_weights`](/Data/Model_weights). 

### Training on AWS
If training is too slow on your local machine, you can use cloud computing services such as AWS. To learn more about training on AWS navigate to [`AWS`](/2_Training/AWS).

This concludes the training step and you are now ready to detect cat faces in new images!