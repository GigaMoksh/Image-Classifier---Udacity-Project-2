import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#Create Argument Parser
parser = argparse.ArgumentParser(
    description='This is a program that predicts flower class',
)

parser.add_argument('image_path', action = "store")
parser.add_argument('saved_model', action = "store")
parser.add_argument('--top_k', action = "store", dest = "top_k", type = int)
parser.add_argument('--category_names', action = "store", dest = "category_names")

#Parse the arguments and set to appropriate variables
results = parser.parse_args()
image_path = results.image_path
saved_model = results.saved_model
category_filename = results.category_names

#If top_k is not provided, set the default to top 5 classes
if results.top_k == None:
    top_k = 5
else:
    top_k = results.top_k

#Define useful functions
def process_image(image):
    ''' Resize and normalize image for MobileNet model
        Arguments
        ---------
        image: image object in numpy array format
        returns resized and normalized image in numpy array format
    '''
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict_class(image_path, model, top_k=5):
    ''' Predict top k flower classes with probabilities
        Arguments
        ---------
        image_path: file path to the image
        model: keras model to predict class
        top_k: number of most probable k classes
        returns k most probable classes along with probabilities in unsorted ndarray format
    '''
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    final_img = np.expand_dims(processed_test_image, axis=0)
    preds = model.predict(final_img)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]
    return probs, classes

#Load model
model = tf.keras.models.load_model(saved_model,
                                  custom_objects={'KerasLayer':hub.KerasLayer})

#Process image and predict its classes and probabilities
image = np.asarray(Image.open(image_path)).squeeze()
probs, classes = predict_class(image_path, model, top_k)

#If category name file is provided, convert the predicted keys into corresponding names
if category_filename != None:
    with open(category_filename, 'r') as f:
        class_names = json.load(f)
    keys = [str(x+1) for x in list(classes)]
    classes = [class_names.get(key) for key in keys]

#Finally, print the results!
print('These are the top {} classes'.format(top_k))
for i in np.arange(top_k):
    print('Class: {}'.format(classes[i]))
    print('Probability: {:.3%}'.format(probs[i]))
    print('\n')