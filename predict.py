import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
from PIL import Image
import warnings
import logging

warnings.filterwarnings('ignore')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(
    description='This is a program that predicts flower class',
)

parser.add_argument('image_path', action="store")
parser.add_argument('saved_model', action="store")
parser.add_argument('--top_k', action="store", dest="top_k", type=int)
parser.add_argument('--category_names', action="store", dest="category_names")

results = parser.parse_args()
image_path = results.image_path
saved_model = results.saved_model
category_filename = results.category_names

# If top_k is not provided, set the default to top 5 classes
top_k = 5 if results.top_k is None else results.top_k


def process_image(image):
    image = tf.image.resize(tf.cast(image, tf.float32),
                            (224, 224))
    image /= 255
    image = image.numpy()
    return image


def predict_class(image_path, model, top_k=5):
    test_image = np.asarray(Image.open(image_path))
    processed_image = process_image(test_image)
    final_img = np.expand_dims(processed_image, axis=0)
    preds = model.predict(final_img)
    probs = - np.partition(-preds[0], top_k)[:top_k]
    classes = np.argpartition(-preds[0], top_k)[:top_k]
    return probs, classes


model = tf.keras.models.load_model(saved_model,
                                   custom_objects={'KerasLayer': hub.KerasLayer})


image = np.asarray(Image.open(image_path)).squeeze()
probs, classes = predict_class(image_path, model, top_k)

if category_filename:
    with open(category_filename, 'r') as f:
        class_names = json.load(f)
    k = [str(i + 1) for i in list(classes)]
    classes = [class_names.get(x) for x in k]

print(f'These are the top {top_k} classes')
for i in np.arange(top_k):
    print(f'Class: {classes[i]}')
    print('Probability: {:.3%}'.format(probs[i]))
    print('\n')
