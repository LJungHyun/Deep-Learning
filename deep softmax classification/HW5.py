import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('image.csv', delimiter=',',dtype=np.float32)
x_data = xy[:1847,0:-1]
y_data = xy[:1847,[-1]]

x_test_data = xy[1848:,0:-1]
y_test_data = xy[1848:,[-1]]

learning_rate = 0.001

X = tf.placeholder(tf.float32, shape = [None, 19])
Y = tf.placeholder(tf.int32, shape = [None, 1])

Y_one_hot = tf.one_hot(Y,7)
Y_one_hot = tf.reshape(Y_one_hot, [ -1, 7])

# Layer1
W1 = tf.get_variable("W1", shape= [19,13],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([13]))
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)

# Layer2
W2 = tf.get_variable("W2",shape=[13,13],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([13]))
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)

# Layer2
W3 = tf.get_variable("W3",shape=[13,13],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([13]))
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3)

W4 = tf.get_variable("W4",shape=[13,7],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([7]))
hypothesis = tf.matmul(L3,W4) + b4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10005):
    cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:
        print(step, "Cost : ", cost_val)

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={ X: x_test_data, Y: y_test_data}))