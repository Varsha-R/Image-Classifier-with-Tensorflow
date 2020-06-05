import argparse
import utilities
import tensorflow_hub as hub
import tensorflow as tf

parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('image_path', type=str, help='Specify path to the image')
parser.add_argument('model', type=str, help='Specify the path to the model to use')
parser.add_argument('--top_k', type=int, help='To return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
args = parser.parse_args()

image_path = args.image_path
model = args.model
top_k = args.top_k
category_names = args.category_names

image = utilities.process_image(image_path)
loaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
probs, classes = utilities.predict(image, loaded_model, top_k)

if category_names != None:
    class_names = utilities.get_class_names(category_names)
    names_class =[]
    for i in classes:
        names_class.append(class_names.get(str(i+1)))
    for i in range(0, len(probs)):
        print("Class:", names_class[i], "\nLabel:", classes[i], "\nProbability:", probs[i], "\n");
else:
    for i in range(0, len(probs)):
        print("Label: ", classes[i], "\nProbability: ", probs[i], "\n");

        