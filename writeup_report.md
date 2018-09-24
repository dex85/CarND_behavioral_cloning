#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./img/nvidia5.jpg "Model Architecture"
[image1]: ./img/model.png "Model Visualization"
[center]: ./img/center.jpg "Center driving"
[left_center]: ./img/left_center.jpg "Center driving left camera"
[right_center]: ./img/right_center.jpg "Center driving right camera"
[center_rev]: ./img/center_rev.jpg "Center reverse driving"
[recovery1]: ./img/recovery_1.jpg "Recovery Image"
[recovery2]: ./img/recovery_2.jpg "Recovery Image"
[recovery3]: ./img/recovery_3.jpg "Recovery Image"
[center_flipped]: ./img/center_flipped.jpg "Flipped Image"
[hsv_random]: ./img/hsv_random.jpg "HSV Changed"
[loss_epochs]: ./img/loss100e.png "Loss over epochs"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* unchanged drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 139,142,145,148,151) as well as five fully connected layers (lines 155-159) with units between 1164 and 1. Each layers utilizes a RELU activation function since it's said to be better for convolutions than sigmoid activation(lectures). It also uses three (max) pooling layers (lines 140,143,146) for downsampling with a 2x2 stride.

The model also includes a lambda layer (line 137) and a cropping layer(line 135). The cropping layer removes the sky/mountains/trees and unnecessary data which is not useful for the model and would cause weaker performance. The lambda layer is for normalizing the pixel from 0-1 and then centered from -.5 to .5 for zero mean.

For reducing overfitting the model uses a dropping layer (line 152) with a keeping rate of 0.5.

####2. Attempts to reduce overfitting in the model

I assumed by the very randomness of the model I wouldn't need a dropout layer... I added one since many other students used it and it's seems to be best practice to reduce overfitting in general (line 152)

Another attempt was to separate and shuffle the data into a training and validation set (line 129) Moreover the generator function (line 45) shuffles (line 51) it's input data as well to further reduce overfitting.

Additionally I adjusted the batch size downwards (256 to 64) since it could lead to overfitting as described in the suggested paper at [https://arxiv.org/pdf/1609.04836.pdf].

Finally the model was tested several times by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 162).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I mainly used he center lane driving approach on the first track an utilized all three cameras for the training. In my opinion given the right and left camera point of views is almost like recovering driving but added recovering scenes because it was recommended.

The data acquisition was done in four six.
* the first trial was center lane driving
* the second trial was center lane driving in reverse
* the third trial consisted of scenes where the car recovers from sides of the road and slowly drives sharp curves
* the fourth trial was careful driving in order to master the curves (which is really difficult sometimes)
* the fifth and sixth trial were like the first and second but on the second track

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to essentially give the network as much as possible different data. I guessed three laps per trial would be enough (except third and fourth). Choose or adjust a well know architecture and make the car clone the "safe" driving.

My first step was to use a convolution neural network model similar to the LeNet Architecture. I thought this model might be appropriate because the instructor used it... Of course, this would have been to easy. Next I thought about the introduced AlexNet, GoogLeNet and ResNet. Well I thought a while but went straight to the NVIDIA introduced structure for autonomous driving. I assumed it's well known, I have some medium sized NVIDIA cards and it was build specifically for the purpose of autonomous driving. I only had to add another fully connected layer with one neuron.
![image0]
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with a ration of 0.2. I actually had good results in numbers, low error in training and validation set. The simulation was not good though. I guessed there is data missing and did more trials as described above.

Then I followed the suggestions from the instructor by first using all three camera pictures. At that point I wrongly assumed that the fourth and fifth column are corresponding steering angles and added the introduced corrections as well. That turned out wrong what the simulation showed later. After that I added the suggested cropping layer and that already gave the car the ability to drive without touching the curbs at all which was again a coincident.

The model only used the data of trial one and two. Adding recovering driving made model performing worse and it drove the car directly into the water or field. A forum entry gave the idea to only use scenes for the purpose of recovery driving. So the third trials consisted of sequences of recovering from the side of the road and special parts of the track. But still the car wouldn't manage to go through. There was a problem with the cv2 since it open in BGR but the drive.py uses RGB. Moreover it was strange that if I not converted the image explicitly to a numpy array the simulation would almost always fail. No matter how many data I gave the model. So, after fixing the code problem I was happy to finally have a working model.

After that I added one more data augmentation and needed to implement a generator(numpy arrays are really big, map to file didn't work well with keras)

The whole process was more like a trial and error to see what impact each modeling step had. After the first working model (which was more an accident really) I failed many times but learned a lot as well.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and not touching the curbs.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Layer (type)           |      Output Shape        |      Param #   
-----------------------|--------------------------|--------------
cropping2d_1 (Cropping2D)    | 75, 320, 3        |    0         
lambda_1 (Lambda)      |      75, 320, 3  |      0         
conv2d_1 (Conv2D)         |   71, 316, 24 |    1824      
max_pooling2d_1 (MaxPooling2x2) |36, 158, 24 |   0         
conv2d_2 (Conv2D)      |      32, 154, 36 |      21636     
max_pooling2d_2 (MaxPooling2x2) | 16, 77, 36 |    0         
conv2d_3 (Conv2D)       |     12, 73, 48  |      43248     
max_pooling2d_3 (MaxPooling2x2) |6, 37, 48   |      0         
conv2d_4 (Conv2D)      |      4, 35, 64   |     27712     
conv2d_5 (Conv2D)      |      2, 33, 64   |      36928     
dropout_1 (Dropout)    |      2, 33, 64   |      0         
flatten_1 (Flatten)    |      4224        |      0         
dense_1 (Dense)        |      1164        |      4917900   
dense_2 (Dense)        |      100         |      116500    
dense_3 (Dense)        |      50          |      5050      
dense_4 (Dense)        |      10          |      510       
dense_5 (Dense)        |      1           |      11        

summary            |  
-------------------|----------------
Total params: | 5,171,319.0
Trainable params: | 5,171,319.0
Non-trainable params: | 0.0

The visualization of the architecture was very convenient to get, "plot_model" from the "keras.utils" library and the method "summary" from the model (lines 170-172)
![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded about three laps on track one using center lane driving in both directions. Here is an example image of center lane driving in forward and reverse direction:

![center][center]      ![center rev][center_rev]

Then followed with scenes to recover from the side of the road.

![recover1][recovery1] ![recover2][recovery2] ![recover3][recovery3]

After that a slow one lap drive on track one. In addition I drove the car on track two three laps in each direction. But I skipped the data from the second track later on, since there wasn't a significant improvement for the test track. It just took more time to train an the model was bigger as well...

To augment the data set, I also flipped images and angles thinking that this would help the model to generalize and make it more robust. For example, here is an image that has then been flipped:

![center][center] ![center flipped][center_flipped]

Another augmentation process which was really helpful is to randomly adjust the V channel after converting the image to HSV picture.

![center][center] ![hsv adjusted][hsv_random]

I also followed the instructions to utilizing all three camera images:

![left center][left_center] ![center][center] ![right center][right_center]

After the collection process, I had  number of data points. I then preprocessed this data by normalizing the data and cropping 25  pixels from the bottom and 60 pixels from the top.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 25 as evidenced by the graph. The test error function is decreasing while the validation error slowly increases which means the model is overfitting after epoch 25

![loss][loss_epochs]

I used the mean-squared error function for evaluating the model and adam optimizer to train the model so that manually adjusting learning rate wasn't necessary.

The training/fitting or the data to the model was realized by using a generator function to provide batches of pictures. That was necessary due limited amount of RAM on GPU and CPU. It was also a good way to provide constantly random data for the model. The generator randomly chose an image from all the trials (center, left, right), or 80% of it, and then randomly flipped, adjusted its V channel or don't augment until one batch was full and handed it to Keras.
