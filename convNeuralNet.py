from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Read in mnist data from official mnist source
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#
# Setup W*x + b = y and the expression to optimize:  cross-entropy
#
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

y = tf.nn.softmax(tf.matmul(x, W) + b, name="softmax")
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    print(i)
    batch = mnist.train.next_batch(50)
    dictFeed = {x: batch[0], y_:batch[1]}

    sess.run(train_step, feed_dict=dictFeed)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    dictFeed = feed_dict={x: mnist.test.images, y_: mnist.test.labels}
    print(sess.run(accuracy, feed_dict=dictFeed))
