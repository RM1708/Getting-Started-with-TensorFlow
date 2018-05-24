
# coding: utf-8

# In[ ]:


#Import libraries for simulation
import tensorflow as tf
import numpy as np
from sympy import *

import matplotlib.pyplot as plt

import IPython.display as display


# In[ ]:


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""

    """Only one set of print output indicates
    this is called only once"""
    
    a = np.asarray(a)
    print("a.ndim: {}; a.shape: {}".format( a.ndim, a.shape))
    print("a: ", a)

    a = a.reshape(list(a.shape) + [1,1])

    print("a.ndim: {}; a.shape: {}".format( a.ndim, a.shape))
    print("a.ndim", a.ndim)
    print("a: ", a)
    
    """NOTE: Returned as tf.constant"""
    return tf.constant(a, dtype=1)


# In[ ]:


def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
    """laplce_k is a tf.constant.
    Is that WHY the call is not made? See comments in make_kernel"""
    return simple_conv(x, laplace_k)


# In[ ]:



N = 500


# In[ ]:


# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)


# In[ ]:


# Some rain drops hit a pond at random points
#for n in range(100 * 2):
#  a,b = np.random.randint(0, N, 2)
#  u_init[a,b] = np.random.uniform(low=0, high=10)

a = N/2; b = N/2
SideOfBrick = 20
Delta_PlusMinus = int(SideOfBrick/2)


# In[ ]:


for i in range(-Delta_PlusMinus, Delta_PlusMinus, 1):
    for j in range(-Delta_PlusMinus, Delta_PlusMinus, 1):
        x = int(a + i); y = int(b + j)
        u_init[x, y] = 100


# In[ ]:


plt.imshow(u_init)
plt.show()


# In[ ]:


# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)
# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())


# In[ ]:


# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))


# In[ ]:


sess = tf.InteractiveSession()
# Initialize state to initial conditions
tf.global_variables_initializer().run()


# In[ ]:


# Run 1000 steps of PDE
for i in range(1000):
  # Step simulation
    step.run({eps: 0.03, damping: 0.04})
        # Visualize every 5 steps
    if i % 10 == 0:
        display.clear_output()
        print("i:{}".format(i))
        plt.imshow(U.eval())
        plt.show()

print


# In[ ]:


sess.close()

