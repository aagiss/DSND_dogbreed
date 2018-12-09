# DSND_dogbreed
Use Deep Learning and Keras to classify dog breeds or which breed a human face looks like

### Table of Contents

1. [Project Overview](https://github.com/aagiss/DSND_dogbreed#overview)
2. [Project Statement](https://github.com/aagiss/DSND_dogbreed#statement)
3. [Metrics](https://github.com/aagiss/DSND_dogbreed#metrics)
4. [Data Exploration](https://github.com/aagiss/DSND_dogbreed#dataexploration)
5. [Data Visualization](https://github.com/aagiss/DSND_dogbreed#datavisualization)
6. [Methodology](https://github.com/aagiss/DSND_dogbreed#methodology)
7. [Results](https://github.com/aagiss/DSND_dogbreed#results)
8. [Conclusion](https://github.com/aagiss/DSND_dogbreed#conclusion)
8. [Installation](https://github.com/aagiss/DSND_dogbreed#installation)
9. [Running the code](https://github.com/aagiss/DSND_dogbreed#running)
10. [Licensing, Authors, and Acknowledgements](https://github.com/aagiss/DSND_dogbreed#licensing)

<a href="https://aa-dogbreed.herokuapp.com"><img src="https://github.com/aagiss/DSND_dogbreed/raw/master/screenshot.jpg"></a>

## Project Overview <a name="overview"></a>

This project is the capstone project of the Udacity Data Scientist Nanodegree program. 

An image is provided and if its a dog we will predict dog breed or if it is a human face we will provide a prediction of the dog breed it looks most like. 

A web app is available at [https://aa-dogbreed.herokuapp.com](https://aa-dogbreed.herokuapp.com)

This is a task of image classification. Image Classification has changed over the past few years through the introduction of Convolutional Neural Networks (i.e. CNN). In this project we use CNNs. Although we classify dog breeds the same techniques can be used for any image classification task provided that we are given an appropriate training dataset.

In this project the dog breed training dataset is provided by Udacity and is NOT shared in this repository.


## Project Statement <a name="statement"></a>

In this project we try to detect humans or dogs and predict their most likelly dog breed.

There are 3 main steps in the process
1. Detection of Dogs in an image
2. Detection of Humans in an image
3. Dog Breed classification (provided that a dog or human was detected)

Deep Learning and specifically CNNs can and are used for (1) and (3). Human detection can be done as face detection with traditional Computer Vision algorithms. Alternatively, deep learning could be used for whole human (and not just face) detection as well provided that there is a dataset available. 

With recent advancments in Image Processing we did expect to have high accuracy on this task. Moreover, predicting a dog breed for humans is a fun part of the project thus we developed an application available to anyone online.

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

We also provide a confusion matrix:
<img src="https://github.com/aagiss/DSND_dogbreed/raw/master/confusion_matrix.png">
Three types of errors have 3 or more errors in the testing set (and correspond to the color-lighted non-diagonal spots in the image).
<pre>
True:American_eskimo_dog Predicted:Pomeranian Count:3
True:Finnish_spitz Predicted:Icelandic_sheepdog Count:3
True:Lowchen Predicted:Havanese Count:3
</pre>
Manual inspection of images of these breeds show that it is hard even for humans to make a distinction from an image.



Unfortunatelly, we did not have a good data set for evaluating human face detection and dog detection. Nevertheless, using a dataset of dogs and a dataset of humans (also provided by Udacity) we were able to predict almost perfectly dog images as ones containing dogs and human images as not containing dogs. Human face detection was a bit less accurate with perfect detection of human faces in the dataset, but 13% of the dog images detected as containing faces (as we explain in the following section that was not wrong though).

## Data Exploration <a name="dataexploration"></a>

In the dataset provided by Udacity for dog breed classification:
<pre>
There are 133 total dog categories.
There are 8351 total dog images.

There are 6680 training dog images.
There are 835 validation dog images.
There are 836 test dog images
</pre>

Manual inspection of the images revieled that some images contain humans along with dogs.
Arround 1 of out of 10 images includes a human face, some times posing and some time in unusuall angles.

## Data Visualization <a name="datavisualization"></a>

Since this is essentially a classification task, the main visualizations concern metrics.
Nevertheless, in the following plot the number of samples per breed in the training set are presented.

<img src="https://github.com/aagiss/DSND_dogbreed/raw/master/samples_per_breed_train.png">


## Methodology <a name="methodology"></a>

As described above: There are 3 main steps in the process
1. Detection of Dogs in an image
2. Detection of Humans in an image
3. Dog Breed classification (provided that a dog or human was detected)

Deep Learning and specifically CNNs are used for (1) and (3). Human detection is done using facedetection from the OpenCV library.
<b>Data Preprocessing</b> of the images consist of transforming images to tensors. This includes resizing and normalizing pixel values. 

Regarding <b>Implementation,</b> ResNet50 and Xception CNNs are used for the deep learning image classification. Specifically for detecting dogs in images ResNet50 is run as trained for with the publically available ImageNet training set. This returns a category out of about 1000 possible categories for the image. If that category represents a dog we do assume a dog is present in the image. 

On the other hand, for dog breed classification we do not use Xception directly. Rather we use transfer learning. In other words we use a pretrained version of Xception but do replace the final dense layer (which was predicting ImageNet categories) with a different one that predicts dog breeds. This layer (and only this layer) is then trained on a dataset provided by Udacity.

The choice of the Xception network was done through <b>refinement</b> of the results. Initially we used ResNet50 for both dog detection and breed classification. Nevertheless, after testing with all available networks we concluded that Xception produced the best results for our case.

## Results <a name="results"></a>

As mentioned above dog detection had nearly perfect accuracy on the testing dataset. However, during manual experiments with random images from the internet we came accross some false positives where the image represented a cat.

Face detection had perfect recall on our limited tests, nevertheless some false positives have been encountered.
This was hard to evaluate with the given dataset as some dog images actually contained human faces.

The main process of the project though was dog breed classification and in that task we achieved 84% accuracy among the 133 classes, which we consider is a good result, leading to a usable system.

Experimental validation with randomly selected images through the [app](https://aa-dogbreed.herokuapp.com) validated these results.

## Conclusion <a name="conclusion"></a>

Image Classification has become an easy task by using transfer learning and some state of the art CNNs. In this project we used out of the shelf image processing modules and used transfer learning and the Xception network for our main task i.e. dog breed classification.

One point, that was hard to justify was why the Xception network outperformed all other candidates such as inception-v3 we used. 
On imagenet, reports from various sources do report better accuracy for Xception than Inception V3. 
Nevertheless the difference is minor (<1% or less) compared to our results (5% difference).

Looking at the training logs of the two networks we saw that in Inception, validation loss stopped improving as soon as the 2nd epoch, although training loss continued to decline and validation accuracy did actually improved. I think this strange phenomenon can be attributed to the following: probability assigned by the softmax output of the network is initially consetrated on few classes, however as the model is trained the values of the non-max class increase and although the max class is the correct one probability is spread in such way that crossentropy descreases. We verified this intuition by adding 3 softmax output layers in the output (inception git branch). Adding a softmax output after a previous softmax output helps reduce the entropy of the result. When we did that difference in acuraccy between networks, dropped from 5% to 3%.

One aspect of the implementation that can be improved is about human detection. We do accuratelly detect human faces, however detecting whole human figures would be more challenging and usefull for the concept of this app. OpenCV does not include an out of shelf module for that task and developing one would require lots of training data we did not have access to. Thus, we chose to let this part for a future implementation.

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python, the following must be installed:
<pre>
conda install -c menpo opencv
conda install -c conda-forge keras
conda install -c anaconda pillow
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
conda install -c anaconda flask
conda install -c anaconda requests
</pre>

Alternatively, pip installing the same packages using <pre>pip install -r requirements.txt</pre> and running the code using Python versions 3.* should produce no issues.

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
