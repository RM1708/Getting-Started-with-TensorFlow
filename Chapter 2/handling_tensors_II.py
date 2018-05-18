#handling tensors second example


#STEP 1 --- PREPARE THE DATA
import matplotlib.image as mp_image
filename = "packt.jpeg"
input_image = mp_image.imread(filename)

#dimension
print ('input dim = {}'.format(input_image.ndim))
#shape
print ('input shape = {}'.format(input_image.shape))

#print("input_image[0, 1]: ", input_image[0, 1])
#print("input_image[0][1]: ", input_image[0][1])
#
#print("input_image[0, 2]: ", input_image[0, 2])
#print("input_image[0][2]: ", input_image[0][2])
#
print("input_image[ 2, : 20, :]: ", input_image[ 2, : 20, :])
height,width,depth= input_image.shape

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

import tensorflow as tf

x = tf.Variable(input_image,name='x')


init_vars = tf.global_variables_initializer()

with tf.Session() as session:
    x_transpose = tf.transpose(x, perm=[1, 0, 2])
    session.run(init_vars)
    result=session.run(x_transpose)
    SliceOfInputImage = (session.run(x))[ 2, : 20, :]
    print("Slice of tensor x[ 2, : 20, :] from within session: ", SliceOfInputImage)

print("Slice of tensor x[ 2, : 20, :] from outside session: ", SliceOfInputImage)

plt.imshow(result)

fig = plt.gcf()

plt.show()

fig.savefig("packt_transpose.pdf")
