## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1.summary of the data set:

You can download the dara form this [link](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)

I used the numpy library and len function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

 Using Pandas and seaborn, I plot the dataset category count as below:
 
 <img src="images/data_set_dis.png" width="820" height="248" />
 

 I made an exploratory visualization of the data set showing a sample Image and It's related values 

<img src="images/1.png" width="820" height="248" />

for more Illustration for the dataset I plot the first 50 images with It's label after shuffling the data:

<img src="images/dataset.png" width="820" height="248" />


### Design and Test a Model Architecture

#### 1.  preprocessed the image data. 
I have shuffle the data.

then convert the data to gray scale to make the model foucs on the shape feature instead of the color:

then normalized the image data so that the data has mean zero and equal variance.


 <img src="images/normalized.png" width="820" height="248" />



#### 2.  final model architecture looks like:

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 color image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    |  1x1 stride, same padding, outputs 10x10x16.      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten
| Fully connected		| outputs 400x120.        									|
| RELU					|												|
| Fully connected		| outputs 120x84.        									|
| RELU					|												|
| Fully connected		| outputs 184x43.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3.  trained the model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used using this hyperpramters:

EPOCHS #=100

BATCH_SIZE #=128

loss = cross_entropy

optimizer =adam

learning rate=0.003

#### 4. test result.

My final model results were:

* validation set accuracy of 96% 

* test set accuracy of 93%


 I have choose this model based on alot of experments:
 
 I have started with one Convolution layer and 2 Fully connected layers and the validation set accuracy was about 70%
 by adding the second Convolution and the 3rd connected layers the model was able to have the 93% accuracy with learning rate=0.001,EPOCHS=20
 
 by changing the learning rate=0.003,EPOCHS=100 I got validation set accuracy of 96% and test set accuracy of 93%
 
  <img src="images/model_result.png" width="820" height="248" />
 
 The hyper-parameters tuning:
 
 learning rate=0.001 -->validation set accuracy of 93% ,test set accuracy of 90%
 learning rate=0.003 -->validation set accuracy of 96.4% ,test set accuracy of 93%
 
 EPOCHS=20 -->validation set accuracy of 91.4% ,test set accuracy of 90%
 EPOCHS=100 -->validation set accuracy of 96.4% ,test set accuracy of 93%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="test_images/1.jpg" width="820" height="248" />

<img src="test_images/2.jpg" width="820" height="248" />

<img src="test_images/3.jpg" width="820" height="248" />

<img src="test_images/4.jpg" width="820" height="248" />

<img src="test_images/5.jpg" width="820" height="248" />


after converting to gray and make the normlization 

<img src="images/images_test.png" width="820" height="248" />
#### 2. the model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield       		|    		Go straight or left							| 
| Speed limit (50km/h     			| Speed limit (30km/h) 										|
| Road work					| Priority road											|
| Children crossing	      		| 	Children crossing				 				|
| Turn right ahead			|     Ahead only							|
| Stop			|    Speed limit (60km/h)							|

The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 16.7%. 


#### 3. how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the the  images,

****0****
true label Yield:

predictions:

Go straight or left: 99.91%
Yield: 0.09%
Keep right: 0.00%
Speed limit (50km/h): 0.00%
Speed limit (30km/h): 0.00%

****1****
true label Speed limit (50km/h):

predictions:

Speed limit (30km/h): 81.23%
Speed limit (70km/h): 18.77%
Speed limit (50km/h): 0.00%
Speed limit (80km/h): 0.00%
Wild animals crossing: 0.00%

****2****
true label Road work:

predictions:

Priority road: 84.70%
No entry: 15.29%
Speed limit (30km/h): 0.00%
Speed limit (60km/h): 0.00%
End of no passing: 0.00%

****3****
true label Children crossing:

predictions:

Children crossing: 100.00%
Traffic signals: 0.00%
Beware of ice/snow: 0.00%
Road narrows on the right: 0.00%
Slippery road: 0.00%

****4****
true label Turn right ahead:

predictions:

Ahead only: 99.89%
Turn left ahead: 0.11%
No passing: 0.00%
Yield: 0.00%
Priority road: 0.00%

****5****
true label Stop:

predictions:

Speed limit (60km/h): 99.99%
Turn left ahead: 0.01%
End of no passing: 0.00%
No passing: 0.00%
Priority road: 0.00%





