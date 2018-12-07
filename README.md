# DSND_dogbreed
Use Deep Learning and Keras to classify dog breeds or which breed a human face looks like

### Table of Contents

1. [Project Overview](https://github.com/aagiss/DSND_dogbreed#overview)
2. [Project Statement](https://github.com/aagiss/DSND_dogbreed#statement)
3. [Metrics](https://github.com/aagiss/DSND_dogbreed#metrics)
4. [Running the code](https://github.com/aagiss/DSND_dogbreed#running)
5. [Licensing, Authors, and Acknowledgements](https://github.com/aagiss/DSND_dogbreed#licensing)


## Project Overview <a name="overview"></a>

This project is the capstone project of the Udacity Data Scientist Nanodegree program. 

An image is provided and if its a dog we will predict dog breed or if it is a human face we will provide a prediction of the dog breed it looks most like. 

A web app is available [here]()

This is a task of image classification. Image Classification has changed over the past few years through the introduction of Convolutional Neural Networks (i.e. CNN). In this project we use CNNs. Although we classify dog breeds the same techniques can be used for any image classification task provided that we are given an appropriate training dataset.

In this project the dog breed training dataset is provided by Udacity and is NOT shared in this repository.


## Project Statement <a name="statement"></a>

There are 3 main steps in the process
1. Detection of Dogs in an image
2. Detection of Humans in an image
3. Dog Breed classification (provided that a dog or human was detected)

Deep Learning and specifically CNNs are used for (1) and (3). Human detection is done using facedetection from the OpenCV library.

ResNet50 and Xception CNNs are used for the deep learning image classification. Specifically for detecting dogs in images ResNet50 is run as trained for with the publically available ImageNet training set. This returns a category out of about 1000 possible categories for the image. If that category represents a dog we do assume a dog is present in the image. 

On the other hand, for dog breed classification we do not use Xception directly. Rather we use transfer learning. In other words we use a pretrained version of Xception but do replace the final dense layer (which was predicting ImageNet categories) with a different one that predicts dog breeds. This layer (and only this layer) is then trained on a dataset provided by Udacity.



## Metrics <a name="metrics"></a>

In order to evaluate classification performance we have kept aside a test segment of the Udacity dog breed data.
In this test dataset we do evaluate accuracy of the predicted dog breed. 
In other words, we predict the breed of all dog images in the Udacity dog breed test data and compare predictions to manually suplied tags.
Accuracy varies according to the network we use for transfer learning.
We tested ResNet50, VGG19, Inception and Xception. ResNet50 and Xception prerformed the best (80% and 84% accuracy). Thus we chose to continue with Xception.

To sum up, given that an image has been correctly identified as having a dog we have measured:
<pre>
84% Accuracy on dog breed classification
</pre>

Unfortunatelly, we did not have a good data set for evaluating human face detection and dog detection. Nevertheless, using a dataset of dogs and a dataset of humans (also provided by Udacity) we were able to predict almost perfectly dog images as ones containing dogs and human images as not containing dogs. Human face detection was a bit less accurate with perfect detection of human faces in the dataset, but 13% of the dog images detected as containing faces.

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python, the following must be installed:
<pre>
conda install -c menpo opencv
conda install -c conda-forge keras
conda install -c anaconda pillow
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
</pre>

Pip installing the same packages and running the code using Python versions 3.* should produce no issues.


## File Descriptions <a name="files"></a>

 
<pre>
- data # data used for this task and ETL code
  |- bottleneck_features  # precalculated features from VGG19, ResNet50, Inception, Exception 
  |- dog_images  # dataset provided by Udacity for dog breed classification
- models # the Machine Learning Pipeline used
  |- breed_classes.json	# category index to dog breed
  |- breed_classifier.py # breed classification
  |- breed_classifier_train.py # the training script for breed classification
  |- breed_classifier.model # the weights of the Dense NN classifier
  |- face_detector.py # the OpenCV face detector
  |- haarcascade_frontalface_alt.xml # the pre-trained face detector classifier
  |- dog_detector.py # the dog detector script using ResNet50
  |- cnn_common.py # common functions for CNN training or predicting
- README.md
</pre>


## Running the code <a name="running"></a>

* Face Detector: <pre>cd models; python face_detector.py IMAGE_FILENAME</pre>
* Dog Detector: <pre>cd models; python dog_detector.py IMAGE_FILENAME</pre>
* Breed Classifier: <pre>cd models; python breed_classifier.py IMAGE_FILENAME</pre>

## Licensing, Authors, Acknowledgements <a name="licensing"></a>

Must give credit to Udacity the training data.  Otherwise, feel free to use the code here as you would like!
