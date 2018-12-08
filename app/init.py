import sys
import os
sys.path.append(os.path.join(os.path.split(__file__)[0], '../models' ))
import dog_detector
import face_detector 
import breed_classifier

img_name = os.path.join(os.path.split(__file__)[0], '..', 'test_images', 'dalmatian-puppy.jpg')
print(dog_detector.dog_detector(img_name))
print(face_detector.face_detector(img_name))
print(breed_classifier.breed_classifier(img_name))
