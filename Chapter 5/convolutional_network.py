"""
Zaccone, Giancarlo. "Getting Started with TensorFlow", Chapter 5

The CNN use three basic ideas: 
    local receptive fields, 
    convolution, 
    and pooling.
    
LOCAL RECEPTIVE FIELD
---------------------
Each neuron of the first subsequent layer(the hidden layer immediately following the input layer)
connects to only some of the input neurons. This region is called local receptive field. 
See the figure at Kindle Locations 2046-2048

CONVOLUTION
-----------
For an image of 28x28 pixels in the input layer, if we scan the image (as in a raster scan - no interleaving :-)
with a 5x5 local receptive field, we will get 24x24 neurons in the immediately following hidden layer. 
See (Kindle Location 2061). 

Each neuron, (in the hidden layer following the input layer), thus has *** a *** bias 
AND a set of 25 (5x5) weights connected to its region.

The ***same set of weights and biases *** are used *** for all 24x24 neurons ***. 
This means that *** all *** the neurons in the first hidden layer will recognize 
*** the same features ***, just placed differently in the input image. 
For this reason, *** the map of connections *** from the input layer to the hidden feature map is 
called *** shared weights *** and 
*** bias *** is called *** shared bias ***.

***A complete *** convolutional layer is made from multiple feature maps.

 A great advantage in the sharing of weights and bias is the significant reduction 
 of the parameters involved in a convolutional network. 
 Considering our example, for each feature map we need 25 weights (5x5) and a bias (shared); 
 that is 26 parameters in total. Assuming we have 20 feature maps, we will have 520 parameters to be defined.
 
    *****************************************
    From my note in the book:
    Each neuron in a given feature map has the same set of weights and bias. 
    Each neuron in a feature map recieves the convolved output from a 5*5 cell from the preceding layer 
    (in this case the input layer). That is 25 weights. Plus there is a bias. 
    So, since all the neurons in a feature map have the same set of weights + bias, 
    there are just 26 parameters ***per feature map***.
    ******************************************

 With a fully connected network, with 784 input neurons and, for example, 30 hidden layer neurons, 
 we need 30 more 784x30 bias weights, reaching a total of 23.550 parameters.
 
    *****************************************
     From my note in the book
    (28*28 = 784 inputs) * (30 neurons in the hidden layer)
    = 23520 weights. Add 1 bias weight for each of the neurons. The total comes to 23,550 parameters
    
    Compare this to convolution with 20 feature maps. The latter requires just 520 for 20 feature maps 
    ****************************************

POOLING
------
Pooling layers, are layers immediately positioned after the convolutional layers; 
these ***simplify*** the output information of the previous layer to it (the convolution). 
It takes the input feature maps coming out of the convolutional layer and prepares a condensed feature map. 
For example, we can say that the pooling layer could be summed up, in all its units, 
in a 2x2 region of neurons of the previous layer.

    *****************************************
    From my notes in the book 
    The 24 * 24 outputs from each feature map is aggregated by summing(averaging)
    2*2 contiguous square regions across the 24 * 24 outputs. 
    This aggregated ouput is fed to the next layer - called a pooling layer. 
    The pooling layer thus requires 12*12 neurons per fdeature map 
    *****************************************
Pooling is applied individually to ***each*** feature map.

From the input layer to the second hidden layer 
if we have three feature maps of size 24x24 for the first hidden layer, 
then the second hidden layer will be of size 12x12, since we are assuming that 
for every unit summarize a 2x2 region. 

Combining these three ideas, we form a complete convolutional network.

SUMMARY
------
There are the 28x28 input neurons followed by a convolutional layer 
with a local receptive field 5x5 and assuming 3 feature maps, as a result 
a hidden layer of neurons 3x24x24. 
Then there is the max-pooling applied to 2x2 on the 3 regions of feature maps 
getting a hidden layer 3x12x12. 
The last layer is fully connected: it connects all the neurons of the max-pooling layer 
(in thepresent case that would be 3 * 144 = 432 neurons) to ***all the 10 output neurons***, 
useful to recognize the corresponding output. 
    *****************************************
    My note
    All 10 output neurons. But the output layer shows a 100 neurons ( 10 * 10)?
    Is it that if there are 10 output classes, the output gives 10 probabilities(?) for each class?
    *****************************************    
    
This network will then be trained by gradient descent and the back propagation algorithm.

"""

# Import MINST data
#import input_data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/home/rm/tmp/data/', one_hot=True)
NoOfTrainingImages = mnist.train.num_examples

import tensorflow as tf

# =============================================================================
#def conv2d(img, w, b):
#    conv_result = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME')
#    conv_result_with_bias = tf.nn.bias_add(conv_result, b)
#    rect_out = tf.nn.relu(conv_result_with_bias)
#    return(rect_out)

def max_pool(img, k):
    return tf.nn.max_pool(img, \
                          ksize=[1, k, k, 1],\
                          strides=[1, k, k, 1],\
                          padding='SAME')
# =============================================================================

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# =============================================================================
# CREATE MODEL
# =============================================================================

# Create variables for weights & biases of the layers
# =============================================================================
# the convolutional layer is composed of 32 feature maps.
# See Getting Started with TensorFlow (Kindle Location 2124). 
# =============================================================================
   
#wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32])) # 5x5 conv, 1 input, 32 outputs
#wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64])) # 5x5 conv, 32 inputs, 64 outputs
##wd1 = tf.Variable(tf.random_normal([4*4*64, 1024])) # fully connected, 7*7*64 inputs, 1024 outputs
#wd1 = tf.Variable(tf.random_normal([7*7*64, 1024])) # fully connected, 7*7*64 inputs, 1024 outputs
#                                        #7*7 needed as now switched padding to
#                                        #'SAME'
#wout = tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)

wc1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)) # 5x5 conv, 1 input, 32 outputs
wc2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)) # 5x5 conv, 32 inputs, 64 outputs
#wd1 = tf.Variable(tf.random_normal([4*4*64, 1024])) # fully connected, 7*7*64 inputs, 1024 outputs
wd1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)) # fully connected, 7*7*64 inputs, 1024 outputs
                                        #7*7 needed as now switched padding to
                                        #'SAME'
wout = tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1)) # 1024 inputs, 10 outputs (class prediction)

#replacing random_normal with truncated_normal and specifying the stddev 
#improved the performance to resemble that of mnist_cnn.py.
#The initialization is now as done in layers.py for mnist_cnn.py.
#The change below further improves the performance

#bc1 = tf.Variable(tf.random_normal([32]))
#bc2 = tf.Variable(tf.random_normal([64]))
#bd1 = tf.Variable(tf.random_normal([1024]))
#bout = tf.Variable(tf.random_normal([n_classes]))

initial = tf.constant(0.1, shape=[32])
bc1 = tf.Variable(initial)
initial = tf.constant(0.1, shape=[64])
bc2 = tf.Variable(initial)
initial = tf.constant(0.1, shape=[1024])
bd1 = tf.Variable(initial)
initial = tf.constant(0.1, shape=[n_classes])
bout = tf.Variable(initial)


# Construct model
_X = tf.reshape(x, shape=[-1, 28, 28, 1])
# The following statement was tried to fix a problem which
# I am unable to reproduce. Keeping the statement but not using it.
_X_Tensor = tf.convert_to_tensor(_X, dtype=tf.float32 )

# Convolution Layer1
#conv_Layer1 = conv2d(_X,wc1,bc1)

conv_result_1 = tf.nn.conv2d(_X, wc1, strides=[1, 1, 1, 1], padding='SAME')
#NOTE: Change to 'SAME' needed the asserts to be changed. That is expected

conv_result_with_bias_1 = tf.nn.bias_add(conv_result_1, bc1)
conv_Layer1 = tf.nn.relu(conv_result_with_bias_1)

# Max Pooling (down-sampling)
pooled_layer1 = max_pool(conv_Layer1, k=2)

# Apply Dropout
Out_Layer1 = tf.nn.dropout(pooled_layer1,keep_prob)

# =============================================================================
# CHECK SHAPES OF INTERMEDIATE STAGES (EDGES OF THE GRAPH) OF THE FIRST LAYER
# =============================================================================

# Initializing the variables
init = tf.global_variables_initializer()

# Parameter
BATCH_SIZE = 128

with tf.Session() as sess:
     sess.run(init)
     batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

#     print("x.shape: {}".format(x.shape))
#     print("batch_xs.shape: {}".format(batch_xs.shape))
# =============================================================================
# _X = tf.reshape(x, shape=[-1, 28, 28, 1])
     x1 = sess.run(_X, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(28 == x1.shape[1] and \
            28 == x1.shape[2] and \
            1 == x1.shape[3])

# Running _X was resulting in an Exception with the following calls to sess.run().
# Not able to reproduce the problem. 
# The check below check the shape of _X_Tensor.
# Using _X_Tensor, in place of _X is the fix that was tried.
# The fix worked sucessfully. But there does not
# appear to be anything to fix:-)
# The check is retained. NOTE that _X_Tensor knows its shape[0]
# whereas _X does not know the size of its shape[0]
#
# _X_Tensor = tf.convert_to_tensor(_X, dtype=tf.float32 )
     x1 = sess.run(_X_Tensor, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(28 == x1.shape[1] and \
            28 == x1.shape[2] and \
            1 == x1.shape[3])
# =============================================================================
#conv_result_1 = tf.nn.conv2d(_X, wc1, strides=[1, 1, 1, 1], padding='SAME')
     x1 = sess.run(conv_result_1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(28 == x1.shape[1] and \
            28 == x1.shape[2] and \
            32 == x1.shape[3])
# =============================================================================
#conv_Layer1 = tf.nn.relu(conv_result_with_bias_1)
     x1 = sess.run(conv_Layer1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(28 == x1.shape[1] and \
            28 == x1.shape[2] and \
            32 == x1.shape[3])
# =============================================================================
#pooled_layer1 = tf.nn.max_pool(conv_Layer1, \
#                                  ksize=[1, 2, 2, 1],\
#                                  strides=[1, 2, 2, 1],\
#                                  padding='SAME')
     x1 = sess.run(pooled_layer1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(14 == x1.shape[1] and \
            14 == x1.shape[2] and \
            32 == x1.shape[3])
# =============================================================================
#Out_Layer1 = tf.nn.dropout(pooled_layer1,keep_prob)
#This is after applying dropout. Since drop-out has been loaded as 1.0, 
#the shape remains the same as the output
#from the preceding node. 
     x1 = sess.run(Out_Layer1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(14 == x1.shape[1] and \
            14 == x1.shape[2] and \
            32 == x1.shape[3])
# =============================================================================
     
# NOW CONVOLUTION LAYER 2
#conv_Layer2 = conv2d(Out_Layer1,wc2,bc2)

conv_result_2 = tf.nn.conv2d(Out_Layer1, wc2, strides=[1, 1, 1, 1], padding='SAME')
conv_result_with_bias_2 = tf.nn.bias_add(conv_result_2, bc2)
conv_Layer2 = tf.nn.relu(conv_result_with_bias_2)

# Max Pooling (down-sampling)
pooled_layer2 = max_pool(conv_Layer2, k=2)

# Apply Dropout
Out_layer2 = tf.nn.dropout(pooled_layer2, keep_prob)

# =============================================================================
# CHECK SHAPES OF INTERMEDIATE STAGES (EDGES OF THE GRAPH) OF THE SECOND LAYER
# =============================================================================
#
# Initializing the variables
init = tf.global_variables_initializer()

# Parameter
BATCH_SIZE = 128

with tf.Session() as sess:
     sess.run(init)
     batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

     x1 = sess.run(conv_result_2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(14 == x1.shape[1] and \
            14== x1.shape[2] and \
            64 == x1.shape[3])
# =============================================================================
     x1 = sess.run(conv_Layer2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(14 == x1.shape[1] and \
            14 == x1.shape[2] and \
            64 == x1.shape[3])
# =============================================================================
     x1 = sess.run(pooled_layer2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(7 == x1.shape[1] and \
            7 == x1.shape[2] and \
            64 == x1.shape[3])
# =============================================================================
     x1 = sess.run(Out_layer2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(7 == x1.shape[1] and \
            7 == x1.shape[2] and \
            64 == x1.shape[3])
# =============================================================================

# Fully connected layer
dense1 = tf.reshape(Out_layer2, [-1, wd1.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
activation_relu = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) # Relu activation
Thinned_OP = tf.nn.dropout(activation_relu, keep_prob) # Apply Dropout

# Output, class prediction
pred = tf.add(tf.matmul(Thinned_OP, wout), bout)

# =============================================================================
# CHECK SHAPES OF INTERMEDIATE STAGES (EDGES OF THE GRAPH) OF OUTPUT
# =============================================================================

# Initializing the variables
init = tf.global_variables_initializer()

# Parameter
BATCH_SIZE = 128

with tf.Session() as sess:
     sess.run(init)
     batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

     x1 = sess.run(dense1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(3136 == x1.shape[1])
# =============================================================================
     x1 = sess.run(Thinned_OP, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(1024 == x1.shape[1])
# =============================================================================
     x1 = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
     assert(BATCH_SIZE == x1.shape[0])
     assert(10 == x1.shape[1])
# =============================================================================

mnist.train.reset_counts()

# Parameter
learning_rate = 1.0e-03  #0.001
# Define loss and optimizer
#cost = tf.reduce_mean(\
#              tf.nn.softmax_cross_entropy_with_logits_v2(\
#                                             logits=pred, labels=y))

#See foot notes 5 Learning TensorFlow: A Guide to Building Deep Learning Systems
# (Kindle Locations 516-518)
#"As of TensorFlow 1.0 this is also contained in tf.losses.softmax_cross_entropy." 
cost = tf.losses.softmax_cross_entropy(
    onehot_labels=y,
    logits=pred)

optimizer = tf.train.AdamOptimizer(\
               learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#See foot notes 6 Learning TensorFlow: A Guide to Building Deep Learning Systems 
#(Kindle Locations 516-518)
#"As of TensorFlow 1.0 this is also contained in tf.metrics.accuracy."

#accuracy, acc_op = tf.metrics.accuracy(labels=tf.argmax(y,1), \
#                                  predictions=tf.argmax(pred,1))

#However, the above is tricky to use. It also needs the ssess.run 
#params to be changed
#See: https://stackoverflow.com/questions/46409626/how-to-properly-use-tf-metrics-accuracy
# *****ALSO: http://ronny.rest/blog/post_2017_09_11_tf_metrics/
# 
# Parameters
BATCH_SIZE = 50
display_step = 100
training_iters = 5000 #NoOfTrainingImages//BATCH_SIZE

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    iter = 0
    # Keep training until reach max iterations
    while iter  < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if iter % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(iter) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
			", Training Accuracy= " + "{:.5f}".format(acc))
        iter += 1
    print ("Optimization Finished! After iter: {}".format(iter))
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, \
                                         feed_dict={x: mnist.test.images[:1000], \
                                                    y: mnist.test.labels[:1000], \
                                                    keep_prob: 1.}))
    #From mnist_cnn
    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    import numpy as np
    test_accuracy = np.mean(
        [sess.run(accuracy, feed_dict={x: X[i], \
                                       y: Y[i], \
                                       keep_prob: 1.0}) for i in range(10)])
    print("test accuracy: {}".format(test_accuracy))
