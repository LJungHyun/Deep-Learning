import tensorflow as tf
import matplotlib.pyplot as plt

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
for i in range(10, 30):
    feed_W = i*0.1
    for j in range (0,20):
        feed_b = j*0.1
        curr_cost, curr_W, curr_b = sess.run([cost, W,b], feed_dict={W:feed_W, b:feed_b})
        W_val.append(curr_W)
        cost_val.append(curr_cost)
        b_val.append(curr_b)
        print("current cost:", curr_cost, "current weight:", curr_W, "current bias:",curr_b)
plt.plot(W_val, cost_val)
plt.show()