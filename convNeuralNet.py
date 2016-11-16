import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

# Create a tf weight variable with a small amount of noise.
# Truncated normal is truncated at 2 std deviations.
# Default mean is 0.0
def weight_variable(shape, name=""):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

# Create a bias with slightly positive lean.
def bias_variable(shape, name=""):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
# Setup placeholders for input layer and expected output
#
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")


# Second Layer (first hidden layer)
cNeuronsLyr2 = 30
lyr2_W = weight_variable([784, cNeuronsLyr2], name="lyr2_W")
lyr2_b = bias_variable([cNeuronsLyr2], name="lyr2_b")

lyr2_Activation = tf.nn.sigmoid(tf.matmul(x, lyr2_W) + lyr2_b)

# Third Layer (Output Layer)
lyr3_W = weight_variable([cNeuronsLyr2, 10], name="lyr3_W")
lyr3_b = bias_variable([10], name="lyr3_b")

output = tf.nn.sigmoid(tf.matmul(lyr2_Activation, lyr3_W) + lyr3_b)

# Cross-entropy calc might look a little odd because we're doing (1 - y_)... calc.
# But, this is because cross-entropy must be calc'd over a probability distribution,
# and raw sigmoid activation (which is what is in the "output" tensor) isn't a prob.
# distribution.  Hence, we interpret each neuron as a Bernoulli distribution (0 or 1)
# with output telling us probability of a single neuron in output is a 1.  Doing so
# makes the cross-entropy optimization behave.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output) + (1 - y_) * tf.log(1 - output), reduction_indices=[1]))
tf.scalar_summary("x-entr", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))

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
citemsInSet = 100
iTestSetLim = min(citemsInSet, len(mnist.test.images))
dictTestData = {x: mnist.test.images[0:iTestSetLim], y_: mnist.test.labels[0:iTestSetLim]}

for i in range(100):
    batch = mnist.train.next_batch(citemsInSet)
    if ((i % 10) == 0):
        print("step %d"%(i))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    train_summary = sess.run(mrg_summary, feed_dict={x: batch[0], y_: batch[1]})
    train_summary_writer.add_summary(train_summary, i)
    test_summary = sess.run(mrg_summary, feed_dict=dictTestData)
    test_summary_writer.add_summary(test_summary, i)

print("test accuracy %g"%sess.run(accuracy, feed_dict=dictTestData))

print("Total Elapsed Time:  {0} msec".format(MsecNow() - msecStart))
