# TrainYourOwnYOLO: Training on AWS

If your local machine does not have a GPU, training could take a very long time. To speed things up use an AWS GPU instance. 

## Spinning up a GPU Instance
To spin up a GPU instance, go to **EC2** and select **Launch Instance**. Then go to **Deep Learning AMI (Ubuntu) Version xx.x** and hit **Select**. Under instance type, select **p2.xlarge** as the Instance Type. Proceed by hitting **Review and Launch**. 

![Deep_Learning_AMI](/2_Training/AWS/Screen_Recordings/AWS_Deep_Learning_AMI.gif)

## Starting the Training
Connect to your instance and follow the same steps as on your local machine. Make sure that all your Source Images are in [`TrainYourOwnYOLO/Data/Source_Images`](/Data/Source_Images) and that both 
 - [`TrainYourOwnYOLO/Data/Source_Images/vott-csv-export/data_train.txt`](/Data/Source_Images/vott-csv-export/data_train.txt) and 
 - [`TrainYourOwnYOLO/Model_Weights/data_classes.txt`](/Data/Model_Weights/data_classes.txt) 

are up-to-date.