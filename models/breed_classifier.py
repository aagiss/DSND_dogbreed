import sys
import os
import json
import numpy as np
from cnn_common import path_to_tensor
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from cnn_common import paths_to_tensor
from breed_classifier_train import build_model
from keras.applications.xception import Xception, preprocess_input
base_model = Xception(weights='imagenet', include_top=False)
with open(os.path.join(os.path.split(__file__)[0],'breed_classes.json')) as g:
    classes = json.load(g)
model = build_model((7, 7, 2048))
model.load_weights(os.path.join(os.path.split(__file__)[0],'breed_classifier.model'))

def breed_classifier(img_path):
    """
    This function takes an image path and returns the predicted breed

    Args:
        img_path: path to the image file

    Returns:
        The detected breed
    """
    tensor = path_to_tensor(img_path)
    features = base_model.predict(preprocess_input(tensor))
    return classes[np.argmax(model.predict(features))]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s IMG_PATH' % sys.argv[0])
        sys.exit(1)
    breed = breed_classifier(sys.argv[1])
    print('Breed "%s" detected in "%s"' % (breed, sys.argv[1]))
