import tensorflow as tf
import matplotlib.pyplot as plt

#We take a square ABCD of side R.
#Take one of the vertices - say A - as the center of a circle of unit radius.
#Draw the arc that connect vertices B & D.
# The area of the arc segment is (PI * R^2)/4.
#The area of the square is R^2.
#So if we generate random points that are uniformly distributed
#over the complete square, the number that can be expectd to fall
#within the arc is (PI/4) * Total Number of Points
#PI = NumberWithinTheArc * 4/ TotalNumberOfPoints

trials = 500
hits = 0
hits_ = 0
pi = [] #This list will hold the progressive computed estimate of PI
pi_ = [] 
x = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)
y = tf.random_uniform([1],minval=0,maxval=1,dtype=tf.float32)

sess = tf.Session()
with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            #Check if the point lies within the arc of radius 1
            if x.eval()**2 + y.eval()**2 < 1 :
                hits = hits + 1
                pi.append(( 4 * float( hits) / i)/ trials)
#        var0 = 4 * hits
#        var0 = var0/i
#        var0 /= trials
#        print(var0)
#        print (((4 * hits) / i)/trials)  

    for NoOfTries in range(1,trials * trials):
        #Check if the point lies within the arc of radius 1
        if x.eval()**2 + y.eval()**2 < 1 :
            hits_ = hits_ + 1
            pi_.append(( 4 * float( hits_)) / NoOfTries)


print(len(pi))
print(len(pi_))

print(pi[len(pi) - 1])
print(pi_[len(pi_) - 1])

plt.plot( pi) 
plt.show()

plt.plot( pi_) 
plt.show()




