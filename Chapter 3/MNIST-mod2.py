
# coding: utf-8
"""
This .py file is converted from MNIST-mod2.ipynb and cleaned up.
The objective is to convert it to a module that  
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# https://stackoverflow.com/questions/33664651/import-input-data-mnist-tensorflow-not-working
# See answer by Kongsea
# also answer May 24 at 12:07 by Ataxia
# 

import sys, os, importlib

pyfilepath = "/home/rm/./tensorflow_models/official/mnist/dataset.py"
dirname, basename = os.path.split(pyfilepath) # 

sys.path.append(dirname)
#print("path after append: \n", sys.path)

module_name = os.path.splitext(basename)[0] # /home/rm/./tensorflow_models/official/mnist/dataset.py --> dataset
module_read_data = importlib.import_module(module_name) # name space of defined module 
                                            #(otherwise we would literally look for "module_name")
sys.path.pop()
#print("sys path after pop: \n", sys.path)


# # Imported Functions
# Use (module_read_data) to get the contents of the imported module.

# # Get Images & Data
# 1. Use the imported function, dataset, to get training images and the corresponding labels.
# 2. dataset has been modified by me. Instead of returning a Zip object. It now returns the two MapDataset objects which were being zipped and returned.
# 3. The two objects can be iterated



images, labels = module_read_data.dataset("MNIST_data","train-images-idx3-ubyte", "train-labels-idx1-ubyte")

#print(type(images).__name__)
#print(type(labels).__name__)
#print("No of Images: \n", len(images))
#print("No of labels: \n", len(labels))
#print(images); print(labels)

#Based on https://www.tensorflow.org/programmers_guide/datasets
#See under Creating an iterator
image_iter = images.make_one_shot_iterator()
label_iter = labels.make_one_shot_iterator()

next_image = image_iter.get_next()
next_label = label_iter.get_next()

sess = tf.Session()
#I want to see the images and labels that are at indices 100 thru 109
#Since a one-shot iterator is used it has to be started from its initial position
#Hence we skip the indices 0 thru 99. Read and display from 100 thru 109
for i in range(110):
    if(i < 100):
        image = sess.run(next_image)
        label = sess.run(next_label)
        continue
    image = sess.run(next_image)
    label = sess.run(next_label)
    
#    print("Image Shape: ",image.shape,"\n")
#    print("Image Dim: ",image.ndim,"\n")
#    print("Image: ",image,"\n\n\n")
#    print("Label Shape: ",label.shape,"\n")
    print("i: ", i, "; Label: ", label,"\n\n\n")
    
    image = np.reshape( image,[ 28,28]) 
    plt.imshow( image) 
    plt.show()
    
sess.close()

