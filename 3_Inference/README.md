# TrainYourOwnYOLO: Inference
In this step, we test our detector on images located in [`TrainYourOwnYOLO/Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images).

## Test the Detector
To detect objects run the detector script.
```
python Detector.py
```
The outputs are save to [`TrainYourOwnYOLO/Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). The outputs include the original images with bounding boxes and confidence scores as well as a file called [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) containing the image file paths and the bounding box coordinates. 

### That's all!
Congratulations on training and using your own custom YOLOv3 computer vision application.