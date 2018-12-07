"""
Module to detect a dog in an image
"""
import sys
import numpy as np
from cnn_common import path_to_tensor
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input

ResNet50_model = ResNet50(weights="imagenet")

def dog_detector(img_path):
    """
    This function takes an image path and returns True if a dog is detected in it

    Args:
        img_path: path to the image file

    Returns:
        The number of faces detected in the image
    """
    tensor = path_to_tensor(img_path)
    img = preprocess_input(tensor)
    prediction = np.argmax(ResNet50_model.predict(img))
    return (prediction <= 268) & (prediction >= 151)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s IMG_PATH' % sys.argv[0])
        sys.exit(1)
    has_dog = dog_detector(sys.argv[1])
    if has_dog:
        print('A dog was detected in "%s"' % (sys.argv[1]))
    else:
        print('No dogs were detected in "%s"' % (sys.argv[1]))
