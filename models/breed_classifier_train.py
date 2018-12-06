import sys
import os
import glob
import json
import numpy as np
from cnn_common import paths_to_tensor
from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def load():
    train_files, train_targets = load_dataset(os.path.join(os.path.split(__file__)[0],
                                                '../data/dog_images/train'))
    valid_files, valid_targets = load_dataset(os.path.join(os.path.split(__file__)[0],
                                                '../data/dog_images/valid'))
    test_files, test_targets = load_dataset(os.path.join(os.path.split(__file__)[0],
                                                '../data/dog_images/test'))

    dog_names = [item[29:-1] for item in sorted(glob.glob(os.path.join(os.path.split(__file__)[0],
                                                '../data/dog_images/train/*/')))]

    return {
            'train':{
                     'targets':train_targets
                    }, 
            'valid':{
                     'targets':valid_targets
                    }, 
            'test':{
                    'targets':test_targets
                   }, 
            'classes': dog_names
           }

def load_bottleneck_features(network):
    bottleneck_features = np.load(os.path.join(os.path.split(__file__)[0],
                                 '../data/bottleneck_features/Dog%sData.npz' % network))
    return bottleneck_features

def build_model(input_shape):
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))
    model.add(Dense(133, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def train(model, train_features, train_target, valid_features, valid_target, epochs = 20):
    checkpointer = ModelCheckpoint(filepath=os.path.join(os.path.split(__file__)[0], 'breed_classifier.model'),
                                  verbose=1, save_best_only=True)

    model.fit(train_features, train_target,
          validation_data=(valid_features, valid_target),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

def eval_test_accuracy(model, test_features, test_target):
    model.load_weights(os.path.join(os.path.split(__file__)[0],'breed_classifier.model'))
    predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_features]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_target, axis=1))/len(predictions)
    return test_accuracy

def main():
    data = load()
    with open(os.path.join(os.path.split(__file__)[0], 'breed_classes.json'), 'w') as g:
        g.write(json.dumps(data['classes']))
    bottleneck_features = load_bottleneck_features('Xception')
    model = build_model(bottleneck_features['train'].shape[1:])
    train(model, bottleneck_features['train'], data['train']['targets'], 
          bottleneck_features['valid'], data['valid']['targets'])
    accuracy = eval_test_accuracy(model, bottleneck_features['test'], 
         data['test']['targets'])
    print('='*80)
    print('==', accuracy)
    print('='*80)
    
if __name__ == '__main__':
    main()
