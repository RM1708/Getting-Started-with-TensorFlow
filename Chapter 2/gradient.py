import tensorflow as tf
#import numpy as np

#x_vals = np.array([1, 2, 3])
x_vals = [1, 2, 3]
print("\nInput x values" + \
      " (note: integers):\n{}".\
      format(x_vals))

x_tensor = tf.placeholder(tf.float32)

func =  2*x_tensor*x_tensor   
gradient_tensor = tf.gradients(func, x_tensor)

with tf.Session() as session:
    gradient_vals = session.run(gradient_tensor,\
                                feed_dict={x_tensor:x_vals})
    print("\nGradients:\n{}".format(gradient_vals))
    print("\nGradients[0]:\n{}".format(gradient_vals[0]))
    print("\nGradients[0][0]:\n{}".format(gradient_vals[0][0]))
    print("\nGradients[0][2]:\n{}".format(gradient_vals[0][2]))
    
    print("\nGradients - No Of Elements:\n{}".format(len(gradient_vals)))
    print("\nGradients[0] - No Of Elements:\n{}".\
          format(len(gradient_vals[0])))


