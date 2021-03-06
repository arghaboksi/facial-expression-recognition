# Facial Expression Recognition 
A CNN based facial expression recognition system that performs recognition on real-time videos. 
Classifies each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy,4=Neutral, 5=Sad,6=Surprise).

## Files
1) FER_2013.ipynb - Contains code pertaining to training of the CNN model

2) localization_recognition.py - Identifies faces in a particular frame and performs expression recognition using the already trained model

3) model.json - Stores the model architecture information

4) model.py - Loads the model architecture along with pretrained weights, makes it available to localization_recognition script

5) model_weights.h5 - Pretrained weights


## Dataset 

The Facial Expression Recognition 2013 (FER-2013) Dataset
, Originator:Pierre-Luc Carrier and Aaron Courville

## Packages Used

1) NumPy


2) Matplotlib

3) Tensorflow(Keras) : for training the CNN model that performs expression recognition

4) OpenCV : for localization of faces across a particular frame

## Performance 

Tested the model both on video clips loaded from disk as well as webcam. Some screenshots of video frames - 


![](Screenshots/Capture_1.PNG)

![](Screenshots/Capture_2.PNG)

![](Screenshots/Capture_3.PNG)

![](Screenshots/Capture_4.PNG)

![](Screenshots/Capture_5.PNG)

![](Screenshots/Capture_6.PNG)

![](Screenshots/Capture_7.PNG)

![](Screenshots/Capture_8.PNG)

![](Screenshots/Capture_9.PNG)

![](Screenshots/Capture_10.PNG)

