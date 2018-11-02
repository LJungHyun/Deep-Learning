import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) +b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 200
batch_size = 100

epochs_val = []
accuracy_val1 = []
accuracy_val2 = []
accuracy_val3 = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xc, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict= {X: batch_xc, Y:batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
        epochs_val.append(epoch)
        accuracy_val1.append(accuracy.eval(session=sess, feed_dict= {X: mnist.test.images, Y: mnist.test.labels}))

a = np.array(epochs_val)
c1 = np.array(accuracy_val1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xc, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict= {X: batch_xc, Y:batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
        epochs_val.append(epoch)
        accuracy_val2.append(accuracy.eval(session=sess, feed_dict= {X: mnist.test.images, Y: mnist.test.labels}))

c2 = np.array(accuracy_val2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xc, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict= {X: batch_xc, Y:batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
        epochs_val.append(epoch)
        accuracy_val3.append(accuracy.eval(session=sess, feed_dict= {X: mnist.test.images, Y: mnist.test.labels}))

c3 = np.array(accuracy_val3)

print(c1)
print(c2)
print(c3)

plt.plot(a, c1, 'r', label = '0.01')
plt.plot(a, c2, 'g', label = '0.1')
plt.plot(a, c3, 'b', label = '0.5')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='upper right')
plt.show()

    # print("Learning finished")
    #
    # print("Accuracy: ", accuracy.eval(session=sess, feed_dict= {X: mnist.test.images, Y: mnist.test.labels}))
    #
    # r = random.randint(0, mnist.test.num_examples -1)
    # print("Lbael : ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    # print("Preddiction: ", sess.run(tf.argmax(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))
    #
    # plt.imshow(mnist.test.images[r:r +1].reshape(28,28), cmap='Greys',interpolation='nearest')
    # plt.show()