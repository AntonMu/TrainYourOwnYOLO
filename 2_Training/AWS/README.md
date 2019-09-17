# TrainYourOwnYOLO: Training on AWS

If your local machine does not have a GPU, training could take a very long time. To speed things up use an AWS GPU instance. 

## Spinning up a GPU Instance
To spin up a GPU instance, go to **EC2** and select **Launch Instance**. Then go to **Deep Learning AMI (Ubuntu) Version xx.x ** and hit **Select**. Under instance type, select **p2.xlarge** as the Instance Type. Proceed by hitting **Review and Launch**. 

![Deep_Learning_AMI](/2_Training/AWS/Screenshots/AWS_Deep_Learning_AMI.gif)

## Start the Training
Connect to your instance and follow the same steps as on your local machine. Make sure to copy the up-to-date `data_train.txt` file to the [`vott-csv-export`](/Data/Source_Images/vott-csv-export) folder and the `classes.txt` file to the [`Data/Model_Weights`](/Data/Model_Weights/) folder.



Before getting started we need to download the pre-trained YOLOv3 weights and convert them to the keras format. To run both steps execute:

```
python Download_YOLO_weights.py
```
## Train YOLOv3 Detector
To start the training run the training script:
```
python Train_YOLO.py 
```
Depending on your set-up this process can take a few minutes to a few hours. I recommend using a GPU to speed up training. The final weights are saved in [`Data/Model_weights`](/Data/Model_weights). 

### Training on AWS
If training is too slow on your local machine, you can speed up the training by using a GPU cluster on AWS. To learn more about training on AWS navigate to the [`AWS`](/2_Training/AWS) folder.

This concludes the training step and you are now ready to detect cat faces in new images!