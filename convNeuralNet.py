import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Create a tf weight variable with a small amount of noise.
# Truncated normal is truncated at 2 std deviations.
# Default mean is 0.0
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# Create a bias with slightly positive lean.
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def MsecNow():
    return int(round(time.time() * 1000))

msecStart = MsecNow()

# Here's where all the tensorflow logs will go.
# For example, things like graph viz and learning information
# will be dumped here
logs_path = "/tmp/tensorflow_logs/mnistConvol"


# Read in mnist data from official mnist source
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#
# Setup W*x + b = y and the expression to optimize:  cross-entropy
#
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

# First convoluational vars
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional vars
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
tf.scalar_summary("x-entr", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary("accuracy", accuracy)
#
# Now, it's time to create session and train things.
#
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Setup a summary writer so we can visualize the learning process
mrg_summary = tf.merge_all_summaries() # Get tensor that represents all summaries to make eval easier
train_summary_writer = tf.train.SummaryWriter(logs_path + "/train", graph=tf.get_default_graph())
test_summary_writer = tf.train.SummaryWriter(logs_path + "/test", graph=tf.get_default_graph())

# Setup test set
iTestSetLim = min(10, len(mnist.test.images))
dictTestData = {x: mnist.test.images[0:iTestSetLim], y_: mnist.test.labels[0:iTestSetLim], keep_prob: 1.0}

for i in range(300):
    batch = mnist.train.next_batch(10)
    if ((i % 10) == 0):
        print("step %d"%(i))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_summary = sess.run(mrg_summary, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    train_summary_writer.add_summary(train_summary, i)
    test_summary = sess.run(mrg_summary, feed_dict=dictTestData)
    test_summary_writer.add_summary(test_summary, i)

print("test accuracy %g"%sess.run(accuracy, feed_dict=dictTestData))

print("Total Elapsed Time:  {0} msec".format(MsecNow() - msecStart))
