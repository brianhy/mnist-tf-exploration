import MnistNeuralNet
import itertools


class RunResults(object):
    def __init__(self):
        self.strCmdLine = ""
        self.strRunName = ""
        self.acc = 0
        self.dmsecTrain = 0
        self.iterN = 0


class RunDesc(object):
    def __init(self):
        self.dictArgs = 0


def DictArgsDefCreate():
    dictT = dict()
    dictT["cNeurons"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_cNeuronsHiddenLyr
    dictT["cEpochs"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_cEpochs
    dictT["l2"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_fltL2RegParam
    dictT["citemsMiniBatch"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_citemsBatch
    dictT["logFolder"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_strLogFolder
    dictT["learnRate"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_fltLrnRate
    return dictT


def StrCmdLineFromRd(rd):
    return "StrCmdLineFromRd"

def StrRunNameFromRd(rd):
    strName = "nN{}.cE{}.cM{}.l2{}.".format(rd.dictArgs["cNeurons"], rd.dictArgs["cEpochs"], rd.dictArgs["citemsMiniBatch"], rd.dictArgs["l2"])
    return strName

def RrInitFromRd(rd, iterT):
    rr = RunResults()
    rr.iterN = iterT
    rr.strCmdLine = StrCmdLineFromRd(rd)
    rr.strRunName = StrRunNameFromRd(rd)
    return rr

def SmnnFromRd(rd):
    smnn = MnistNeuralNet.SigmoidMnistNeuralNet(
                            fltLrnRateIn=rd.dictArgs["learnRate"],
                            cEpochsIn=rd.dictArgs["cEpochs"],
                            citemsBatchIn=rd.dictArgs["citemsMiniBatch"],
                            cNeuronsHiddenLyrIn=rd.dictArgs["cNeurons"],
                            fltL2RegParamIn=rd.dictArgs["l2"],
                            strLogFolderIn=rd.dictArgs["logFolder"])
    return smnn


def StrResultFromRrRd(rr, rd):
    strOut = "{0}, {1}, {2}, {3:0.5f}, {4}, {5}, {6}, {7}, {8:0.5f}\n".format(
                rr.strCmdLine,
                rr.strRunName,
                rr.iterN,
                rr.acc,
                rr.dmsecTrain,
                rd.dictArgs["cNeurons"],
                rd.dictArgs["cEpochs"],
                rd.dictArgs["citemsMiniBatch"],
                rd.dictArgs["l2"])

    return strOut


cIters = 3

rd = RunDesc()
rd.dictArgs = DictArgsDefCreate()

lstRd = list()

#
# Setup RunDesc
#

lstEpochs = [100, 500, 1000, 5000, 10000]
lstMiniBatches = [10, 50, 100, 200, 500]
for cEpochs, citemsMiniBatch in itertools.product(lstEpochs, lstMiniBatches):
    rd = RunDesc()
    rd.dictArgs = DictArgsDefCreate()
    rd.dictArgs["cEpochs"] = cEpochs
    rd.dictArgs["citemsMiniBatch"] = citemsMiniBatch

    lstRd.append(rd)

# Runs for epoch/mini batch variation over cNeurons = 100
for cEpochs, citemsMiniBatch in itertools.product(lstEpochs, lstMiniBatches):
    rd = RunDesc()
    rd.dictArgs = DictArgsDefCreate()
    rd.dictArgs["cNeurons"] = 100
    rd.dictArgs["cEpochs"] = cEpochs
    rd.dictArgs["citemsMiniBatch"] = citemsMiniBatch

    lstRd.append(rd)


# Runs for l2 over epochs
lstL2 = [.00001, .0001, .001, .01, .1]
for l2, cEpochs in itertools.product(lstL2, lstEpochs):
    rd = RunDesc()
    rd.dictArgs = DictArgsDefCreate()
    rd.dictArgs["l2"] = l2
    rd.dictArgs["cEpochs"] = cEpochs
    rd.dictArgs["citemsMiniBatch"] = 100

    lstRd.append(rd)

#
# Train and gather results
#
lstRr = list()
irun = 0
for rd in lstRd:
    for iterT in range(cIters):
        rd.dictArgs["logFolder"] = "/tmp/mnistRuns/" + StrRunNameFromRd(rd) + "-{}".format(iterT)
        rr = RrInitFromRd(rd, iterT)

        # Print Status
        print("[{:0.2f}%] {} / {}; iter {} / {} - {}".format(
                    100 * float(irun*cIters + iterT + 1) / (len(lstRd) * cIters),
                    irun + 1,
                    len(lstRd),
                    iterT + 1,
                    cIters,
                    rd.dictArgs["logFolder"]))

        # Train
        smnn = SmnnFromRd(rd)
        smnn.Train()

        # Gather results
        rr.dmsecTrain = smnn.m_dmsecTrain
        rr.acc = smnn.m_accTest
        lstRr.append((rd, rr))
    irun = irun + 1

#
# Output results to file
#
strOutFile = "/tmp/mnistTrain.csv"
print("")
print("Dumping results to CSV '{}' . . .".format(strOutFile))
fhLog = open(strOutFile, "w+")

strHeader = "CmdLine, RunName, Iter, Accuracy, Dmsec train, cNeurons, cEpochs, miniBatch, l2\n"
fhLog.write(strHeader)
for rd, rr in lstRr:
    strOut = StrResultFromRrRd(rr, rd)
    fhLog.write(strOut)

fhLog.close()
print("Finished dumping results")
