# Import libraries for simulation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
     
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
    
    for i in range(11): 
        step.run()
    #    if((0 == i%5) and (200 >= i)):
    #        print("i: {}; not_diverged: {}".format(i, not_diverged.eval()))
    #        print("\ni: {}; ns: {}".format(i, (ns.eval()).imag))
    #        print("i: {}; c: {}".format(i, c.eval()))
    #        print("\i: {}; The tensor zs:\n{}".format(i, zs.eval()))
#        if(tf.logical_not(sess.run(not_diverged))):
        if(not (sess.run(not_diverged)).any):
            print("\nDiverged at i = {}, ".format(i))
            
    print("\n ns:\n {}".format((ns.eval())[5:10, :]))      
    plt.imshow(ns.eval())
    plt.show()
    
finally:
    sess.close()







