# TrainYourOwnYOLO: Inference
In this step, we test our detector on cat and dog images and videos located in [`TrainYourOwnYOLO/Data/Source_Images/Test_Images`](/Data/Source_Images/Test_Images). If you like to test the detector on your own images or videos, place them in the [`Test_Images`](/Data/Source_Images/Test_Images) folder. 

## Testing Your Detector
To detect objects run the detector script from within the [`TrainYourOwnYOLO/3_Inference`](/3_Inference/) directory:.
```
python Detector.py
```
The outputs are saved to [`TrainYourOwnYOLO/Data/Source_Images/Test_Image_Detection_Results`](/Data/Source_Images/Test_Image_Detection_Results). The outputs include the original images with bounding boxes and confidence scores as well as a file called [`Detection_Results.csv`](/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv) containing the image file paths and the bounding box coordinates. For videos, the output files are videos with bounding boxes and confidence scores. For real-time detection use `python Detector.py --webcam` this will open up your webcam and detect your data classes. To list available command line options run `python Detector.py -h`.

### That's all!
Congratulations on building your own custom YOLOv3 computer vision application.

I hope you enjoyed this tutorial and I hope it helped you get our own computer vision project off the ground:

- Please **star** ⭐ this repo to get notifications on future improvements and
- Please **fork** 🍴 this repo if you like to use it as part of your own project.
