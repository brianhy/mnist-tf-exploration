import sys
import time

from tensorflow.examples.tutorials.mnist import input_data
import argparse as argp

import tensorflow as tf

def MsecNow():
    return int(round(time.time() * 1000))

class SigmoidMnistNeuralNet(object):
    s_fltLrnRate=1.0
    s_cEpochs=1000
    s_citemsBatch=100
    s_cNeuronsHiddenLyr=30
    s_fltL2RegParam=0.0
    s_strLogFolder="/tmp/tensorflow_logs/mnistConvol"

    def __init__(self,
                    fltLrnRateIn=s_fltLrnRate,
                    cEpochsIn=s_cEpochs,
                    citemsBatchIn=s_citemsBatch,
                    cNeuronsHiddenLyrIn=s_cNeuronsHiddenLyr,
                    fltL2RegParamIn=s_fltL2RegParam,
                    strLogFolderIn=s_strLogFolder):
        self.m_fltLrnRate = fltLrnRateIn
        self.m_cEpochs = cEpochsIn
        self.m_citemsBatch = citemsBatchIn
        self.m_cNeuronsLyr2 = cNeuronsHiddenLyrIn
        self.m_fltL2RegParam = fltL2RegParamIn

        # Here's where all the tensorflow logs will go.
        # For example, things like graph viz and learning information
        # will be dumped hereIn
        self.m_strLogFolder = strLogFolderIn


    # Create a tf weight variable with a small amount of noise.
    # Truncated normal is truncated at 2 std deviations.
    # Default mean is 0.0
    @staticmethod
    def weight_variable(shape, name=""):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)


    # Create a bias with slightly positive lean.
    @staticmethod
    def bias_variable(shape, name=""):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def Train(self):
        msecStart = MsecNow()

        # Read in mnist data from official mnist source
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        #
        # Setup placeholders for input layer and expected output
        #
        x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")


        # Second Layer (first hidden layer)
        lyr2_W = SigmoidMnistNeuralNet.weight_variable([784, self.m_cNeuronsLyr2], name="lyr2_W")
        lyr2_b = SigmoidMnistNeuralNet.bias_variable([self.m_cNeuronsLyr2], name="lyr2_b")

        lyr2_Activation = tf.nn.sigmoid(tf.matmul(x, lyr2_W) + lyr2_b)

        # Third Layer (Output Layer)
        lyr3_W = SigmoidMnistNeuralNet.weight_variable([self.m_cNeuronsLyr2, 10], name="lyr3_W")
        lyr3_b = SigmoidMnistNeuralNet.bias_variable([10], name="lyr3_b")

        output = tf.nn.sigmoid(tf.matmul(lyr2_Activation, lyr3_W) + lyr3_b)

        # Cross-entropy calc might look a little odd because we're doing (1 - y_)... calc.
        # But, this is because cross-entropy must be calc'd over a probability distribution,
        # and raw sigmoid activation (which is what is in the "output" tensor) isn't a prob.
        # distribution.  Hence, we interpret each neuron as a Bernoulli distribution (0 or 1)
        # with output telling us probability of a single neuron in output is a 1.  Doing so
        # makes the cross-entropy optimization behave.
        cross_entropy = -tf.reduce_sum(y_ * tf.log(output) + (1 - y_) * tf.log(1 - output), reduction_indices=[1])

        # We're going to add in some L2 regularization across the weights in our layers.
        # This will be Lambda/N * Sum(w^2, across all weights) - where N is items in the
        # mini-batch.
        L2_reg = self.m_fltL2RegParam * (tf.nn.l2_loss(lyr2_W) + tf.nn.l2_loss(lyr3_W))
        tf.scalar_summary("L2_reg", L2_reg)

        cost = tf.reduce_mean(cross_entropy + L2_reg) # Here's where we divide cost by N (# items in mini-batch)
        tf.scalar_summary("cost", cost)

        train_step = tf.train.GradientDescentOptimizer(self.m_fltLrnRate).minimize(cost)
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
        train_summary_writer = tf.train.SummaryWriter(self.m_strLogFolder + "/train", graph=tf.get_default_graph())
        test_summary_writer = tf.train.SummaryWriter(self.m_strLogFolder + "/test", graph=tf.get_default_graph())

        # Setup test set
        dictTestData = {x: mnist.test.images[0:self.m_citemsBatch], y_: mnist.test.labels[0:self.m_citemsBatch]}

        for i in range(self.m_cEpochs):
            batch = mnist.train.next_batch(self.m_citemsBatch)
            if ((i % 10) == 0):
                print("step {}".format(i))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            train_summary = sess.run(mrg_summary, feed_dict={x: batch[0], y_: batch[1]})
            train_summary_writer.add_summary(train_summary, i)
            test_summary = sess.run(mrg_summary, feed_dict=dictTestData)
            test_summary_writer.add_summary(test_summary, i)

        return sess.run(accuracy, feed_dict=dictTestData)


def ParseCmdLine():
    parser = argp.ArgumentParser(description="Learn hand-written digits using Sigmoid network")
    parser.add_argument("-r", "--learningRate", default=SigmoidMnistNeuralNet.s_fltLrnRate, type=float, dest="fltLrnRate",
			help="Learning Rate for Gradient Descent [Def - %(default)s]", metavar="LrnRate")
    parser.add_argument("-e", "--epochs", default=SigmoidMnistNeuralNet.s_cEpochs, type=int, dest="cEpochs",
			help="Number of epochs to train [Def - %(default)s]", metavar="[No. Epochs]")
    parser.add_argument("--log", default=SigmoidMnistNeuralNet.s_strLogFolder, dest="strLogFolder",
			help="Folder where to put logs for tensorboard [Def - %(default)s]", metavar="[Log Folder Path]")
    parser.add_argument("-m", "--miniBatchSize", default=SigmoidMnistNeuralNet.s_citemsBatch, type=int, dest="citemsBatch",
			help="Number of items to run per epoch [Def - %(default)s]", metavar="[mini batch size]")
    parser.add_argument("-n", default=SigmoidMnistNeuralNet.s_cNeuronsHiddenLyr, type=int, dest="cNeuronsHiddenLyr",
			help="Number of neurons to put in the hidden layer [Def - %(default)s]", metavar="N")
    parser.add_argument("--l2", default=SigmoidMnistNeuralNet.s_fltL2RegParam, type=float, dest="fltL2RegParam",
			help="L2 Regularization Parameter, 0 => no L2 regularization [Def - %(default)s]", metavar="[L2 Reg Param]")

    return parser.parse_args()

if (__name__ == "__main__"):
    options = ParseCmdLine()
    fltLrnRate = options.fltLrnRate
    cEpochs = options.cEpochs
    citemsBatch = options.citemsBatch
    cNeuronsLyr2 = options.cNeuronsHiddenLyr
    fltL2RegParam = options.fltL2RegParam

    # Here's where all the tensorflow logs will go.
    # For example, things like graph viz and learning information
    # will be dumped here
    strLogFolder = options.strLogFolder

    smnn = SigmoidMnistNeuralNet(fltLrnRate, cEpochs, citemsBatch, cNeuronsLyr2, fltL2RegParam, strLogFolder)
    accuracy = smnn.Train()

    print("acc = {0:.5f}".format(accuracy))
