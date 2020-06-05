from PIL import Image
import json
import numpy as np
import tensorflow as tf

def process_image(image_path):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    image = tf.cast(test_image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image

def get_class_names(json_path):
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def predict(processed_image, model, top_k):
    ps = model.predict(np.expand_dims(processed_image, axis=0))
    probabilities = ps[0]
    if top_k == None:        
        classes = probabilities.argsort()[::-1][:]
        probs = probabilities[classes]
    else:        
        classes = probabilities.argsort()[::-1][:top_k]
        probs = probabilities[classes]
    return probs, classes