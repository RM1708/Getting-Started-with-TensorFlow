# Import libraries for simulation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#for slicing error message
import sys, os

     
#MANDELBROT SET
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

#JULIA SET
#Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definiamo il punto corrente 
Z = X+1j*Y
c = tf.constant(Z.astype("complex64"))

sess = tf.InteractiveSession()
print ("\nThe tensor c:\n{}".format(c.eval()))
sess.close()

#Initialize the variable that is evaluated at each recursion
zs = tf.Variable(c)

ns = tf.Variable(tf.zeros_like(c, "float32"))

#c = complex(0.0,0.75)
#c = complex(-1.5,-1.5)
try:
    sess = tf.InteractiveSession()
    
    tf.global_variables_initializer().run()
    
    result = zs.eval()
    print ("\nThe tensor zs:\n{}".format(result))
    
    # Compute the new values of z: z^2 + x
    zs_ = zs*zs + c
    
    
    #zs_ = zs*zs - c
    
    # Have we diverged with this new value?
    not_diverged = tf.abs(zs_) < 4
    
    step = tf.group(
      zs.assign(zs_),
      ns.assign_add(tf.cast(not_diverged, "float32")),
      )
    
    for i in range(200): 
        step.run()
        if((0 == i%5) and ((20 >= i) or (180 <= i))):
            print("\ni: {}; not_diverged shape: {}".format(i, \
                  not_diverged.eval().shape))
            print("i: {}; not_diverged[250:255, 250:255]:\n {}".\
                  format(i, \
                  not_diverged.eval()[250:255, 250:255]))
    #        print("\ni: {}; ns: {}".format(i, (ns.eval()).imag))
            print("i: {}; c shape: {}".format(i, c.eval().shape))
            print("i: {}; c[250:255, 250:255]:\n {}".format(i, \
                  c.eval()[250:255, 250:255]))
    #        print("\i: {}; The tensor zs:\n{}".format(i, zs.eval()))
            
    ns_array = ns.eval()
    print("\n ns_array dims: {}".format(ns_array.ndim))     
    print("\n ns_array shape: {}".format(ns_array.shape))     
    print("\n ns_array[5:10, 595:]:\n {}".format(ns_array[5:10, 595:]))  
    np.savetxt("ns_array.csv", ns_array, fmt="%4.2f", delimiter=", ",)
    """
    Examine the ns_array.csv in LibreOffice Calc. The largest value is 11.0.
    Observe the pattern of 11s. It corresponds to the red portion of the plot.
    """
    
    plt.imshow(ns_array, cmap="coolwarm")
    """
    To find out the available color maps:
    Run plt.imshow(ns_array, cmap=""). That will throw an exception.
    The exception is handled below. The error message gives a list of the
    available color maps
    """
    plt.show()
    
except (TypeError, NameError, ValueError) as err:
    print("\nERROR ERROR ERROR:")
    print("------------------- ")
    print(err)
    """
    from:
    https://stackoverflow.com/questions/1278705/python-when-i-catch-an-exception-how-do-i-get-the-type-file-and-line-number
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("\n\tException Type: ",exc_type,"; \n\tIn File: ", \
          fname, "; ", "\n\tAt Line No: ",exc_tb.tb_lineno)  
    
    sess.close() 
else:
    sess.close()
#finally:
#    sess.close()







