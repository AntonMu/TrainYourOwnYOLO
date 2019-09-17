# TrainYourOwnYOLO: Training on AWS

If your local machine does not have a GPU, training could take a very long time. To speed things up use an AWS GPU instance. 

## Spinning up a GPU Instance
To spin up a GPU instance, go to **EC2** and select **Launch Instance**. Then go to **Deep Learning AMI (Ubuntu) Version xx.x ** and hit **Select**. Under instance type, select **p2.xlarge** as the Instance Type. Proceed by hitting **Review and Launch**. 

![Deep_Learning_AMI](/2_Training/AWS/Screenshots/AWS_Deep_Learning_AMI.gif)

## Start the Training
Connect to your instance and follow the same steps as on your local machine. Make sure to copy the up-to-date `data_train.txt` file to the [`vott-csv-export`](/Data/Source_Images/vott-csv-export) folder and the `classes.txt` file to the [`Data/Model_Weights`](/Data/Model_Weights/) folder.