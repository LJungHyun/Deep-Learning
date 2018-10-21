import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data.csv', delimiter=',',dtype=np.float32)
x_data = xy[:8000,0:-1]
y_data = xy[:8000,[-1]]
cost_sum = 0

x_test_data = xy[8001:,0:-1]
y_test_data = xy[8001:,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape = [None, 5])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([5,1]), name = 'weigth')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\n Prediction : \n", hy_val)

i = 0
for  i in range(1999):
    cost_val, hy_val = sess.run([cost, hypothesis], feed_dict={X: [[x_test_data[i][0],x_test_data[i][1],
                                                                    x_test_data[i][2],x_test_data[i][3],
                                                                    x_test_data[i][4]]],Y: [[y_test_data[i][0]]]})
    cost_sum = cost_val + cost_sum
    if i < 5 :
        print("\n\n Real_val : ", y_test_data[i][0], "\nhy_val : \n", hy_val)

print("cost EVR", cost_sum/2000)