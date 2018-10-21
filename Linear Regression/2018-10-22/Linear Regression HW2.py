import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')


X =[1,2,3,4,5,6,7,8,9,10]
Y = [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]

W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
b_val = []
for i in range(0, 40):
    feed_W = i*0.1
    feed_b = 0.5
    curr_cost, curr_W, curr_b = sess.run([cost, W,b], feed_dict={W:feed_W, b:feed_b})
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    b_val.append(curr_b)
    print("current cost:", curr_cost, "current weight:", curr_W, "current bias:",curr_b)

a = np.array(W_val)
c = np.array(b_val)
d = np.array(cost_val)

a = a.reshape(2, 20)
c = c.reshape(2, 20)
d = d.reshape(2, 20)

print("a", a)
print("b", c)
print("d", d)

ax.plot_surface(a,c,d)
ax.set_xlabel('weight')
ax.set_ylabel('bias')
ax.set_zlabel('cost')
ax.set_zlim(0,200)
plt.show()