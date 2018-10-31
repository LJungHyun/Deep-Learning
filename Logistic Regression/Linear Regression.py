import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('magic_train.csv', delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
cost_sum = 0

xy_test = np.loadtxt('magic_test.csv', delimiter=',',dtype=np.float32)

x_test_data = xy_test[:,0:-1]
y_test_data = xy_test[:,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape = [None, 10])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([10,1]), name = 'weigth')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accruracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost : ", cost_val, "\n Prediction : \n", hy_val)

hy_val, correct, accuracy = sess.run([hypothesis,predicted,accruracy], feed_dict={X:x_test_data, Y:y_test_data})

print("\n Hypothesis : ", hy_val, "\nCorrect (Y):", correct, "\nAccuracy : ", accuracy)