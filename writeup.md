# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/visualization.jpg "visualization"
[image2]: ./output_images/dataset_class_count.jpg "class_count"
[image3]: ./test_images/23.jpg "Slippery road"
[image4]: ./test_images/12.png "Priority road"
[image5]: ./test_images/14.jpg "Stop"
[image6]: ./test_images/35.jpg "Ahead only"
[image7]: ./test_images/40.jpg "Roundabout mandatory"



[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

Here is another exploratory visualization of the data set. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did't use any pre-processing techniqu.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 3x3 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64   				|
| Fully connected		| 256 node, outputs 256.      					|
| RELU					|												|
| Fully connected		| 125 node, outputs 125.       					|
| RELU					|												|
| Fully connected		| 43 node, outputs 43.        					|
| RELU					|												|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an 'AdamOptimizer' optimizer, set up the 'LEARNING_RATE_BASE' with 0.001.
I used leraning rate decay technique，set up the 'LEARNING_RATE_DECAY' with 0.999.
I used moving average decay technique, set up the 'LEARNING_RATE_DECAY' wiht 0.99.
I also used dropout techiqu, set up the 'DROPOUT_RATE' with 0.5.
When I training model, the  epochs I used is 200. Every epochs have 100 batches.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were: 0.964766
* training set accuracy of 1.0
* validation set accuracy of 0.982766 
* test set accuracy of 0.964766

The first architecture to try was LeNet because of the model's good performance on the MNIST dataset.

However, this model has a certain underfitting problem, the precision of the model is not high, and the precision will fluctuate up and down when the model is trained to the later stage.

To resolve the model underfitting problem, I added a layer of full connection layer.

As a result, the number of parameters increased greatly and there was over-fitting phenomenon, so I used regularization technology to limit the scale of the model, and meanwhile used dropout technology to randomly discard some parameters, which effectively solved the over-fitting problem.

In order to solve the problem of fluctuation of precision in the later period of training, learning rate attenuation method is used.At the early stage of training, the learning rate is set to be larger. With the increase of training rounds, the learning rate decays to a smaller value, making the accuracy of the model more stable in the later stage.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road		    | Speed limit (80km/h)    			        	|
| Stop		            | Stop		            		             	|
| Slippery road  		| Slippery road  								|
| Ahead only   		    | Ahead only   		    		 				|
| Roundabout mandatory  | Roundabout mandatory        					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

I think this may be due to the different number of signs in each category。

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the fisrt image, the predict is incorrect that this is a speed limit (80km/h) sign, and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.16         			| Speed limit (80km/h)              			| 
| 0.14     				| Speed limit (60km/h) 				        	|
| 0.13					| Yield						                	|
| 0.07	      			| No passing					 				|
| 0.06				    | Speed limit (120km/h)		                	|


For the second image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop              							| 
| 0.0     				| No vehicles 				        			|
| 0.0					| Speed limit (80km/h)							|
| 0.0	      			| Pedestrians					 				|
| 0.0				    | Right-of-way at the next intersection			|







