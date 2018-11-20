import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets("Data/MNIST_data/", one_hot=True)

nb_classes =10

X = tf.placeholder(tf.float32, [None,784])

x_image = tf.reshape(X,[-1,28,28,1],name = 'x_image')
tf.summary.image('x_image', x_image)

Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.name_scope('Layer1') as scope:
    W1 = tf.Variable(tf.random_normal([784, nb_classes]))
    b1 = tf.Variable(tf.random_normal([nb_classes]))
    hypothesis = tf.nn.softmax(tf.matmul(X,W1) + b1)

    W1_hist = tf.summary.histogram('weights1',W1)
    b1_hist = tf.summary.histogram('biases1',b1)
    hypothesis1_hist = tf.summary.histogram('hypothesis1',hypothesis)

with tf.name_scope('Cost') as scope:
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
    cost_summary = tf.summary.scalar('cost',cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.argmax(Y,1))

with tf.name_scope('Accuracy') as scope:
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    accuracy_summary = tf.summary.scalar('Accuracy',accuracy)

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('Logs/mnist.log')
    writer.add_graph(sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost,optimizer], feed_dict={X: batch_xs, Y:batch_ys})

            avg_cost += c/ total_batch

        summary = sess.run(merge_summary, feed_dict= {X:batch_xs, Y:batch_ys})
        writer.add_summary(summary,global_step = epoch)
        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict= {X:mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis,1),feed_dict={X: mnist.test.images[r:r+1]}))
    print("sample image shape:". mnist.test.images[r:r+1].shape())
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Gresys', interpolation='nearest')
    plt.show()