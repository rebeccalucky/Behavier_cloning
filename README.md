# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./home/workspace/training_Images/center_2016_12_01_13_33_06_108.jpg "center camera"
[image2]: ./home/workspace/training_Images/left_2016_12_01_13_33_06_108.jpg "left camera"
[image3]: ./home/workspace/training_Images/right_2016_12_01_13_33_06_108.jpg "right camera"
[image4]: ./home/workspace/training_Images/flip_left.jpg "left camera image flipped"
[image5]: ./home/workspace/training_Images/flip_right.jpg "right camera image flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture includes:
* preprocessing layers: images cropping (line 57, Cropping2D layer) and images normalizing (line 59, Lambda layer).
* Conv2D layer with 5x5 filter and relu activation, followed by MaxPooling2D layer (line 63-65).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 69-72).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 76-79).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 83-86).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 90-93).
* Flatten layer(line 96).
* Two Fully-connected layer with relu activation and followed by Dropout(line 100-106).
* Output layer(line 110).

#### 2. Attempts to reduce overfitting in the model

The model contains several dropout layers in order to reduce overfitting (model.py lines 72,79,86,93,101,106). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 116,117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center camera, left camera and right camera.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use CNN to train the model and avoid overfitting in the process.

My first step was to use a convolution neural network model similar to the LeNet.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set and on the validation set. This implied that the model was underfitting. 

To improve the model, I add more convolutional layers.

This time, the model had a low mean squared error on the training set but high mse on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by addding dropout layers.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 63-110) consisted of a convolution neural network with the following layers and layer sizes:
* Conv2D layer with 5x5 filter and relu activation, followed by MaxPooling2D layer (line 63-65).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 69-72).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 76-79).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 83-86).
* Conv2D layer with 3x3 filter and relu activation, followed by MaxPooling2D layer and Dropout layer(line 90-93).
* Flatten layer(line 96).
* Two Fully-connected layer with relu activation and followed by Dropout(line 100-106).
* Output layer(line 110).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded one laps on track one using center lane driving. Here is an example image of center lane driving from center camera:

![alt text][image1]

I also use left and right camera to show the model how to operate when the car is on the very side of lane. The relatived steering angle minus 0.2 for right camera and add 0.2 for left camera.

![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images of left/right camera and angles thinking that this would enlarge the data set. For example, here is two images that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 40180 number of data points. I then preprocessed this data by cropping and normalization.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.I used an adam optimizer so that manually training the learning rate wasn't necessary.
