import sys
import time

from tensorflow.examples.tutorials.mnist import input_data
import argparse as argp
import numpy as np
import tensorflow as tf
from PIL import Image as pimg
import os

def MsecNow():
    return int(round(time.time() * 1000))


# Save the provided image/width/height to strSaveFilePath
#
# Note:  Will resize picture to 255 for easier visualization
def SaveImage(dataMnistPic, width, height, strSaveFilePath):
    imgMnist = pimg.new("L", (width, height))
    imgMnist.putdata(dataMnistPic, 255)
    imgMnistSave = imgMnist.resize((112, 112))
    imgMnistSave.save(strSaveFilePath)

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
                    strLogFolderIn=s_strLogFolder,
                    fExportPicsOfMislabeledIn=False,
                    fUseFullTestSetIn=False):
        self.m_fltLrnRate = fltLrnRateIn
        self.m_cEpochs = cEpochsIn
        self.m_citemsBatch = citemsBatchIn
        self.m_cNeuronsLyr2 = cNeuronsHiddenLyrIn
        self.m_fltL2RegParam = fltL2RegParamIn
        self.m_fExportPicsOfMislabeled = fExportPicsOfMislabeledIn
        self.m_fUseFullTestSet = fUseFullTestSetIn

        # Here's where all the tensorflow logs will go.
        # For example, things like graph viz and learning information
        # will be dumped hereIn
        self.m_strLogFolder = strLogFolderIn


    # Save picture of each mislabeled input (mislabeled == network got answer wrong)
    #
    #   networkPreds  - Predicted outputs made by the neural net (list)
    #   expectedPreds - Expected (aka Correct) output for given input (list)
    #   inputPixelData - List of pixel data used as input by network
    #   strFolderSave - Folder into which we'll export pictures.
    #
    #   Note:  The expectation is that networkPreds/expectedPreds/inputPixelData are
    #          all the same size
    @staticmethod
    def ExportMislabeledPics(networkPreds, expectedPreds, inputPixelData, strFolderSave):
        os.makedirs(strFolderSave, exist_ok=True) # First, create directory to save into

        lstiWrong = np.where(networkPreds != expectedPreds) # Create list of indexes for wrong predictions
        lstiWrong = lstiWrong[0] # output of np.where is really a list of lists, so unpack that.
        for iWrong in lstiWrong:
            strFilePathSave = os.path.join(strFolderSave, "{}-{}-{}G.bmp".format(iWrong, expectedPreds[iWrong], networkPreds[iWrong]))
            SaveImage(inputPixelData[iWrong], 28, 28, strFilePathSave)


    # Create a tf weight variable with a small amount of noise.
    # Truncated normal is truncated at 2 std deviations.
    # Default mean is 0.0
    @staticmethod
    def weight_variable(shape, name=""):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)


    # Create a bias with slightly positive lean.
    @staticmethod
    def bias_variable(shape, name=""):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


    def Train(self):
        msecStart = MsecNow()

        print("lrnRate={}, "
                "cEpochs={}, "
                "citemsBatch={}, "
                "cNeuronsLyr2={}, "
                "fltL2RegParam={}, "
                "strLogFolder={}, "
                "fExportPicsOfMislabeled={}, "
                "fUseFullTestSet={}".format(self.m_fltLrnRate,
                                            self.m_cEpochs,
                                            self.m_citemsBatch,
                                            self.m_cNeuronsLyr2,
                                            self.m_fltL2RegParam,
                                            self.m_strLogFolder,
                                            self.m_fExportPicsOfMislabeled,
                                            self.m_fUseFullTestSet))

        # Read in mnist data from official mnist source
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        #
        # Setup placeholders for input layer and expected output
        #
        with tf.name_scope("Layer1-Input"):
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")


        # Second Layer (first hidden layer)
        with tf.name_scope("Layer2-Hidden"):
            lyr2_W = SigmoidMnistNeuralNet.weight_variable([784, self.m_cNeuronsLyr2], name="lyr2_W")
            lyr2_b = SigmoidMnistNeuralNet.bias_variable([self.m_cNeuronsLyr2], name="lyr2_b")

            lyr2_Activation = tf.nn.sigmoid(tf.matmul(x, lyr2_W) + lyr2_b, name="lyr2_a")

        # Third Layer (Output Layer)
        with tf.name_scope("Layer3-Output"):
            lyr3_W = SigmoidMnistNeuralNet.weight_variable([self.m_cNeuronsLyr2, 10], name="lyr3_W")
            lyr3_b = SigmoidMnistNeuralNet.bias_variable([10], name="lyr3_b")

            output = tf.nn.sigmoid(tf.matmul(lyr2_Activation, lyr3_W) + lyr3_b, name="lyrOutput")

        # Cross-entropy calc might look a little odd because we're doing (1 - y_)... calc.
        # But, this is because cross-entropy must be calc'd over a probability distribution,
        # and raw sigmoid activation (which is what is in the "output" tensor) isn't a prob.
        # distribution.  Hence, we interpret each neuron as a Bernoulli distribution (0 or 1)
        # with output telling us probability of a single neuron in output is a 1.  Doing so
        # makes the cross-entropy optimization behave.
        with tf.name_scope("Cost"):
            with tf.name_scope("x-entropy"):
                kludge = 1e-10 # Add some kludge to the tf.log() calls so we don't pass in a negative arg
                cross_entropy = -tf.reduce_sum(y_ * tf.log(output + kludge) + (1 - y_) * tf.log(1 - output + kludge), reduction_indices=[1], name="x-entropy-calc")

            # We're going to add in some L2 regularization across the weights in our layers.
            # This will be Lambda/N * Sum(w^2, across all weights) - where N is items in the
            # mini-batch.
            with tf.name_scope("L2_Reg"):
                L2_reg = self.m_fltL2RegParam * (tf.nn.l2_loss(lyr2_W) + tf.nn.l2_loss(lyr3_W))
                L2_reg = tf.identity(L2_reg, "L2_reg-calc")
                tf.scalar_summary("L2_reg", L2_reg)

            cost = tf.reduce_mean(cross_entropy + L2_reg, name="cost-calc") # Here's where we divide cost by N (# items in mini-batch)
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
        citemsTestSet = mnist.test.images.shape[0] if self.m_fUseFullTestSet else self.m_citemsBatch
        dictTestData = {x: mnist.test.images[0:citemsTestSet], y_: mnist.test.labels[0:citemsTestSet]}

        for i in range(self.m_cEpochs):
            batch = mnist.train.next_batch(self.m_citemsBatch)
            dictIn = {x: batch[0], y_: batch[1]}
            sess.run(train_step, feed_dict=dictIn)
            train_summary = sess.run(mrg_summary, feed_dict=dictIn)
            train_summary_writer.add_summary(train_summary, i)
            test_summary = sess.run(mrg_summary, feed_dict=dictTestData)
            test_summary_writer.add_summary(test_summary, i)

        train_summary_writer.flush()
        train_summary_writer.close()
        test_summary_writer.flush()
        test_summary_writer.close()

        self.m_dmsecTrain = MsecNow() - msecStart
        self.m_accTest, networkOutput = sess.run([accuracy, output], feed_dict=dictTestData)

        if (self.m_fExportPicsOfMislabeled):
            networkPreds = np.argmax(networkOutput, 1)
            expectedPreds = np.argmax(dictTestData[y_], 1)
            SigmoidMnistNeuralNet.ExportMislabeledPics(networkPreds, expectedPreds, dictTestData[x], os.path.join(strLogFolder, "MislabeledImgs/"))

        # Close the session and reset the graph when done.
        # If we don't reset the graph, then subsequent calls to train
        # within the same python session will fail.
        sess.close()
        tf.reset_default_graph()

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
    parser.add_argument("-p", action="store_true", dest="fExportPicsOfMislabeled",
			help="Export pictures of those inputs that were incorrect [Def - %(default)s]")
    parser.add_argument("-f", action="store_true", dest="fUseFullTestSet",
			help="When measuring accuracy of network, use full set set (otherwise, use test set that's same size as mini-batch) [Def - %(default)s]")

    return parser.parse_args()

if (__name__ == "__main__"):
    options = ParseCmdLine()
    fltLrnRate = options.fltLrnRate
    cEpochs = options.cEpochs
    citemsBatch = options.citemsBatch
    cNeuronsLyr2 = options.cNeuronsHiddenLyr
    fltL2RegParam = options.fltL2RegParam
    fExportPicsOfMislabeled = options.fExportPicsOfMislabeled
    fUseFullTestSet = options.fUseFullTestSet

    # Here's where all the tensorflow logs will go.
    # For example, things like graph viz and learning information
    # will be dumped here
    strLogFolder = options.strLogFolder

    smnn = SigmoidMnistNeuralNet(fltLrnRate, cEpochs, citemsBatch, cNeuronsLyr2, fltL2RegParam, strLogFolder, fExportPicsOfMislabeled, fUseFullTestSet)
    smnn.Train()

    print("acc = {0:.5f}".format(smnn.m_accTest))
