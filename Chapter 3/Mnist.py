#coding utf-8

import numpy as np
import matplotlib.pyplot as plt

#Using input_data we load the data sets :

# https://stackoverflow.com/questions/33664651/import-input-data-mnist-tensorflow-not-working
# See answer by Kongsea
# Tensor site also has the same. See on https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
mnist_images = input_data.read_data_sets("MNIST_data", one_hot=True)


"""
import input_data

mnist_images = input_data.read_data_sets\
               ("MNIST_data/",\
                one_hot=False)
"""

#train.next_batch(10) returns the first 10 images :
pixels,real_values = mnist_images.train.next_batch(10)

#it also returns two lists, the matrix of the pixels loaded, and the list that contains the real values loaded:

print ("list of values loaded \n",real_values)
print("\nNo Of Values loaded: \n", len(real_values))

print("\nImage Data Array Dims: \n", pixels.ndim)
print("\nImage Data Array Shape: \n", pixels.shape)

example_to_visualize = 5
print ("element {}".format(example_to_visualize + 1),", of the list, ", "(i.e. ", real_values[example_to_visualize], "),"\
                    , " is dispayed\n")

image = pixels[ example_to_visualize,:] 
image = np.reshape( image,[ 28,28]) 
plt.imshow( image) 
plt.show()


