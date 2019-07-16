# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

In this project:

* I am exploring, summarizing and visualizing the traffic signs data set
* Designing, training and testing a model architecture.
* Using the model to make predictions on new traffic signs images.
* Analyzing the softmax probabilities of the new traffic signs images

### Data Set Summary & Exploration

#### 1. A basic summary:

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset:

The code for this step is contained in the 4-7 code cells of the IPython notebook.  

Here is an exploratory visualization of the data set. 

1) The content of the csv file.
1) 50 example images of the German Traffic Signs:

![](http://i.imgur.com/pZqti1F.png)

2) A bar chart showing how many traffic signs are for each class:
![](http://i.imgur.com/E2wnT6q.png)

3) An large example of a traffic sign
![](http://i.imgur.com/Ba01i0F.png)

### Design and Test a Model Architecture

#### Preprocesse of the image data

The code for this step is contained in the 8-18 code cells of the IPython notebook.

As a first step, I decided to add light to the images by adjusting their gamma. I noticed that there are many dark images.

As a second step, I decided to convert the images to grayscale because
it is easier to compare 2 gray images than 2 colorful ones (less channels).

Here are 2 example of a traffic sign image before and after grayscaling.

before:

![](http://i.imgur.com/CcUyVAZ.png)

after:

![](http://i.imgur.com/74RD6zi.png)


before:

![](http://i.imgur.com/Ba01i0F.png)

after:

![](http://i.imgur.com/2llscz7.png)

As a third step, I decided to use Histogram Equalization of the image (cv2.equalizeHist()) which normally improves the contrast of the image.

As the last step, I normalized the image data to further improve the contrast.

#### 2. Setting up training, validation and testing data

The 13-16 code cells of the IPython notebook contains the code for augmenting the data set. 
I decided to generate additional data because I saw that some of the signs were rotated and/or shifted. 
To add more data to the the data set, I used Keras with the following techniques :"Random Rotations" and "Random Shifts".I used them because they add more data to the dataset as described here:"http://machinelearningmastery.com/image-augmentation-deep-learning-keras/"


#### 3. The final model architecture

The code for my final model is located in the 19 cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| dropout				| keep prob : 0.5								|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6
|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 |
| dropout		        | keep prob : 0.5       						|
| Max pooling			| etc.        								
|
| Flatten 				| Input = 5x5x16. Output = 400.					|
| Fully Connected 		| Input = 400. Output = 120.					|
| Sigmoid 		        | 
|
| Fully Connected  		| Input = 120. Output = 84
|
| Sigmoid 		        | 
|
| Fully Connected  		| Input = 84. Output = 43
|

#### 4. Training of the model

The code for training the model is located in the 21-24 cells of the ipython notebook. 

To train the model, I used an AdamOptimizer, a batch size of 128 and 100 epochs.
The hyperparameters I used include learning rate of 0.001. 


#### 5. The approach taken for finding a solution

The code for calculating the accuracy of the model is located in the 19-20 code cells of the Ipython notebook.

My final model results were:
* training set accuracy of 0.974
* validation set accuracy of 0.915 
* test set accuracy of 0.890

If an iterative approach was chosen:
* The first architecture that I tried was LeNet since it is simple, small and effective.
* The problem with the initial architecture was that it was under fitting.
* I adjusted the architecture to by changing the activation functions from relu to dropout and sigmoid functions instead.
* I tuned the parameters of epochs and learning rate. I chose to do more epochs so the network will have more time to learn and lower the learning rate so the network won't over fitting.

### Test a Model on new Images

#### 1. The qualities of new images that might be difficult to classify

I wanted to test my model on new trafic signs images.

First I chose good qulity images with light, good contrast,and without shifts or rotations.
I gave the set of images the name: "Good images"

Here are five good quality images I found on the web:

![](http://i.imgur.com/I76OcUP.png) 
![](http://i.imgur.com/hGE1rsP.png) ![](http://i.imgur.com/thGzavM.png) 
![](http://i.imgur.com/LcGK0UI.png) ![](http://i.imgur.com/ao2oaAg.png)

Second, I chose images with problems:

1) The traffic sign is smudged:
![](http://i.imgur.com/housphs.png)

2) There are 3 traffic signs instead of one:
![](http://i.imgur.com/2vL7Jme.png)

3) The traffic sign is small and rotated:
 ![](http://i.imgur.com/EmN3C0Z.png)
 
4) There is no such sign on the data:
![](http://i.imgur.com/7X5YrKB.png)

5) The traffic sign is rotated:
![](http://i.imgur.com/Eq6pkHp.png)

I gave the set of images the name: "Bad images"

#### 2. The model's predictions on these new traffic signs

The code for making predictions on my final model is located in the 25-38 code cells of the Ipython notebook.

Here are the results of the prediction of the "good" images:

| Image			              |     Prediction	        				                  	| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)  				                    | 
| Speed limit (60km/h)  | No entry     						                           |
| Children crossing	   	| Road work				                                 |
| No entry	      		     | No entry					 	                             		|
| Stop Sign 		         	| Ahead only     			                         			|

The model was able to correctly guess 2 of the 5 "good" traffic signs images, which gives it an accuracy of 40% on good quality images.

Here are the results of the prediction of the "bad" images:

| Image			              |     Prediction	        		                  			| 
|:---------------------:|:---------------------------------------------:| 
| No vehicles           | Roundabout mandatory  				                    | 
| 3 traffic signs       | Turn left ahead     				                      |
| Speed limit (50km/h) 	| General caution				                           |
| Speed limit (130km/h) | Road work					 		                            	|
| Stop Sign 		         	| Stop Sign     					                          	|


The model was able to correctly guess 1 of the 5 "bad" traffic signs images, which gives it an accuracy of 20% on bad quality images.

#### 3. The certainty of the model is when predicting on the new images:

The code for making predictions on my final model is located in the 39-43 code cell of the Ipython notebook.

Here are the bar charts for the 5 "good" images:

# The first good quality image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Speed limit (30km/h)   				| 
| .04     				| Speed limit (50km/h) 					|
| .00					| Speed limit (80km/h)					|
| .00	      			| Speed limit (20km/h)
|
| .00				    | Speed limit (100km/h)      			|


![](http://i.imgur.com/JUrasbU.png)

The model is sure (94%) that the first image is a Speed limit (30km/h) sign and in fact, it is a Speed limit (30km/h) sign.
This prediction looks good, because the model selected Speed limit signs and predicted the correct one.

# The second good quality image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .46         			| No entry   							| 
| .40     				| No passing 							|
| .12					| Slippery road							|
| .01	      			| Turn left ahead					 	|
| .00				    | Children crossing    

![](http://i.imgur.com/CYhXHnj.png)

The model is not so sure (46%) if the second image is a "No entry" sign or a "No passing" sign, and in fact, it is neither, it is Speed limit (60km/h).

# The third good quality image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Road work   							| 
| .02     				| Beware of ice/snow 					|
| .00					| Bicycles crossing						|
| .00	      			| Wild animals crossing					|
| .00				    | Slippery road 

![](http://i.imgur.com/fQPMfWC.png)

The model is sure (95%) that the third image is a "Road work" sign and in fact, it is a "Children crossing" sign,which looks similar if the image inside is not clear.However,this is a good quality image, so this is a problem.

# The fourth good quality image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| No entry   							| 
| .03     				| No passing 							|
| .00					| Stop									|
| .00	      			| Keep right					 		|
| .00				    | Speed limit (60km/h) 

![](http://i.imgur.com/UpwdsUH.png) 

The model is sure (96%) that the fourth image is a "No entry" sign and in fact, it is a "No entry" sign.


# The fifth good quality image


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .63         			| Ahead only   							| 
| .15     				| Turn left ahead 						|
| .15					| Keep right							|
| .03	      			| Stop					 				|
| .00				    | Yield 

![](http://i.imgur.com/SIByaCf.png)

The model is pretty sure (63%) that the fifth image is a "Ahead only" sign and in fact, it is not. It is a "Stop" sign.The model was only able predict in 3% accuracy that this is a "Stop" sign.


Here are the bar charts for the 5 "bad" images:

# The first bad quality image


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .70         			| Roundabout mandatory   				| 
| .16     				| Keep right 							|
| .03					| Traffic signals						|
| .03	      			| Priority road					 		|
| .02				    | Speed limit (30km/h) 


![](http://i.imgur.com/MsYlHCj.png)

In this image the sign is smudged. The model is pretty sure (70%) that the first image is a "Roundabout mandatory" sign and in fact, it is a "No vehicles" sign.


# The second bad quality image


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .76         			| Turn left ahead   					| 
| .22     				| No entry 								|
| .00					| Beware of ice/snow					|
| .00	      			| Keep right					 		|
| .00				    | Speed limit (60km/h) 

![](http://i.imgur.com/sK85ITk.png)

This image contains 3 traffic signs on the same pole. The model is pretty sure (76%) that the second image is a "Turn left ahead" sign and in fact, there are 3 signs.

# The third bad quality image


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61         			| General caution   					| 
| .09     				| Priority road 						|
| .06					| Dangerous curve to the right			|
| .06	      			| Speed limit (80km/h)					|
| .06				    | Children crossing 


![](http://i.imgur.com/ZTywVPt.png)

The third traffic sign is rotated and small. The model is pretty sure (60%) that the third image is a "Road work" sign and in fact, it is a "Speed limit (50km/h)" sign.

# The fourth bad quality image



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| Road work   							| 
| .00     				| Keep right 							|
| .00					| Turn left ahead						|
| .00	      			| Go straight or right					|
| .00				    | Bumpy road 

![](http://i.imgur.com/plpPSFW.png)

The fifth image contains a traffic sign that is not in the data.The model is very sure (90%) that the fourth image is a "Road work" sign and in fact, it is a "Speed limit (130km/h)" sign.

# The fifth bad quality image


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .41         			| Stop   								| 
| .29     				| Turn right ahead 						|
| .25					| No entry							
|
| .01	      			| Yield					 				|
| .00				    | Priority road 


![](http://i.imgur.com/WlpkRqm.png)

The fifth traffic sign is rotated and shifted. The model is pretty sure (40%) that the fifth image is a "Stop" sign and in fact, it is a "Stop" sign.



In general, the model is less sure when it is trying to predict the sign in bad quality images, and even makes a mistake when there is no such sign on the data.
