# Image-Classifier-with-Tensorflow
In this project, the jupyter notebook contains the code for building the image classifier model with Tensorflow.

The command line application includes the predict.py file that uses a trained network to predict the class for an input image. A utilities.py module is used just for utility functions like preprocessing images. 
The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

## Basic usage:

$ python predict.py /path/to/image saved_model

## Options:
--top_k : Return the top KK most likely classes:
$ python predict.py /path/to/image saved_model --top_k KK

--category_names : Path to a JSON file mapping labels to flower names:
$ python predict.py /path/to/image saved_model --category_names map.json

## Examples
For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.
- Basic usage:

$ python predict.py ./test_images/orchid.jpg my_model.h5
- Options:

Return the top 3 most likely classes:

$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3

Use a label_map.json file to map labels to flower names:

$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
