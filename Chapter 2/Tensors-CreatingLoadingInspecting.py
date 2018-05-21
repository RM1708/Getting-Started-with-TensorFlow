"""
For numeric multidimensional tensors, Demonstrate How To
1. Print the contents of a numeric tensor. This would help
in debugging.

2. Print the contents of a placeholder after it is loaded

"""
import numpy as np
#1-D numpy array for loading into tensor
array_1d = np.array([1.3,1,4.0,23.99])

#Check the np array created
#It is 1 dimensional array.4 elements
print ("array_1d:\n", array_1d)
print ("\narray_1d[0]:\n", array_1d[0])
print ("\narray_1d[2]:\n", array_1d[2])

#The following should give
#Array no of Dims, Shape, & data type: 1 (4,) float64
print("\nArray no of Dims, Shape, & data type:", array_1d.ndim, \
      array_1d.shape, array_1d.dtype)

#Now we want to convert the np array into a tensorflow tensor
import tensorflow as tf

#Creates a 1-D tensor from the 1-D np array
tf_tensor = tf.convert_to_tensor(array_1d, dtype=tf.float64)

################################################################
#By creating an interactive session, the contents of the tensor can be checked.

#Check its Shape Type and rank
#Expected output:
#tf_tensor: Shape: (4,), type: <dtype: 'float64'>, &  rank: 1:
#
#Since we still do not have a session we create an interactive session
#It is needed for .eval() to work
sess = tf.InteractiveSession()

print("\ntf_tensor: \nClass: {}; \nShape: {}; \
      \ntype: {}; &  \nrank: {}:".format(  \
      type(tf_tensor.eval()), \
      tf.shape(tf_tensor.eval()), \
      tf_tensor.dtype, \
      (tf.rank(tf_tensor)).eval()))
print ("\n(tf_tensor).eval():", (tf_tensor).eval())
print ("(tf_tensor[0]).eval():", (tf_tensor[0]).eval())
print ("(tf_tensor[1]).eval():", (tf_tensor[2]).eval())

sess.close()
################################################################

with tf.Session() as sess:
#Instead of an interactive one a special session can be launched.
#Now the session does not have to be explicitly closed.
#The session lasts for the scope of the name space
    
# Same result as above
    print("\ntf_tensor: \nClass: {}; \nShape: {}; \
          \ntype: {}; &  \nrank: {}:".format(  \
          type(tf_tensor.eval()), \
          tf.shape(tf_tensor.eval()), \
          tf_tensor.dtype, \
          (tf.rank(tf_tensor)).eval()))

    print ("\nsess.run(tf_tensor):", sess.run(tf_tensor))
    print ("sess.run(tf_tensor[0]):", sess.run(tf_tensor[0]))
    print ("sess.run(tf_tensor[2]):", sess.run(tf_tensor[2]))



# =============================================================================
# Repeat for 2D arrayu
array_2d=np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])
 
print ("\narray_2d:\n", array_2d)
print ("\narray_2d[3][3]:\n", array_2d[3][3])
print ("\narray_2d[0:2,0:2]:\n", array_2d[0:2,0:2])
 
tf_tensor_2d = tf.convert_to_tensor(array_2d, dtype=tf.float64)


#Expected output
#tf_tensor_2d: Shape: (4, 4), type: <dtype: 'float64'>, &  rank: 2:
sess = tf.InteractiveSession()
print("\ntf_tensor_2d: Shape: {}, type: {}, &  rank: {}:".format(  \
      tf_tensor_2d.shape, tf_tensor_2d.dtype, \
      (tf.rank(tf_tensor_2d)).eval()))

sess.close()

with tf.Session() as sess:
    result = sess.run(tf_tensor_2d)
    print ("\ntf_tensor_2d:\n",result)
 
# =============================================================================
#Expt with feeding via placeholder
#
y_placeholder=tf.placeholder(dtype=tf.float64, \
                             shape=tf_tensor_2d.shape, name='y_tensor')
y = tf.identity(y_placeholder)
with tf.Session() as sess:
#     sess.run(y, feed_dict={y_placeholder: tf_tensor_2d})
#     Gives
#     TypeError: The value of a feed cannot be a tf.Tensor object. 
#     Acceptable feed values include Python scalars, strings, lists, 
#     numpy ndarrays, or TensorHandles.For reference, the tensor object 
#     was Tensor("Const_55:0", shape=(4, 4), dtype=float64) which was 
#     passed to the feed with key Tensor("y_tensor_10:0", shape=(4, 4), 
#                                        dtype=float64).
                                        
     result = sess.run(y, feed_dict={y_placeholder: array_2d})
     print ("\nTensor in y_placeholder, after identity op:\n",result)
 
print("\nDONE")