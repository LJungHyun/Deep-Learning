import tensorflow as tf

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 916., 143.]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name = 'weigth1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weigth2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weigth3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b