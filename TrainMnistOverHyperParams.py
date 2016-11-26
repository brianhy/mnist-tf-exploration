import MnistNeuralNet
import itertools
import time
import datetime as dt

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
    dictT["lstcNeuronsPL"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_lstcNeuronsPerLayer
    dictT["cEpochs"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_cEpochs
    dictT["l2"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_fltL2RegParam
    dictT["citemsMiniBatch"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_citemsBatch
    dictT["logFolder"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_strLogFolder
    dictT["learnRate"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_fltLrnRate
    dictT["fUseFullTestSet"] = MnistNeuralNet.SigmoidMnistNeuralNet.s_fUseFullTestSet

    return dictT


def StrCmdLineFromRd(rd):
    return "StrCmdLineFromRd"

def StrRunNameFromRd(rd):
    strName = "nN{}.cE{}.cM{}.l2{}.".format(rd.dictArgs["lstcNeuronsPL"], rd.dictArgs["cEpochs"], rd.dictArgs["citemsMiniBatch"], rd.dictArgs["l2"])
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
                            lstcNeuronsPerLayerIn=rd.dictArgs["lstcNeuronsPL"],
                            fltL2RegParamIn=rd.dictArgs["l2"],
                            strLogFolderIn=rd.dictArgs["logFolder"],
                            fUseFullTestSetIn=rd.dictArgs["fUseFullTestSet"])
    return smnn


def StrResultFromRrRd(rr, rd):
    strOut = "{0}, {1}, {2}, {3:0.5f}, {4}, {5}, {6}, {7}, {8:0.5f}, {9}\n".format(
                rr.strCmdLine,
                rr.strRunName,
                rr.iterN,
                rr.acc,
                rr.dmsecTrain,
                rd.dictArgs["lstcNeuronsPL"],
                rd.dictArgs["cEpochs"],
                rd.dictArgs["citemsMiniBatch"],
                rd.dictArgs["l2"],
                rd.dictArgs["fUseFullTestSet"])

    return strOut


def MsecNow():
    return int(round(time.time() * 1000))


def StrElapsedTimeFromTd(td):
    lstStr = list()
    if (td.days > 0):
        lstStr.append("{} days".format(td.days))
    seconds = td.seconds
    if (seconds > 60 * 60): # Handle hours first
        cHours = seconds // (60 * 60)
        lstStr.append("{} hours".format(cHours))
        seconds -= cHours * 60 * 60
    if (seconds > 60): # Handle mins
        cMins = seconds // 60
        lstStr.append("{} mins".format(cMins))
        seconds -= cMins * 60
    lstStr.append("{} secs".format(seconds))
    lstStr.append("{} mc secs".format(td.microseconds))

    return ", ".join(lstStr)

cIters = 3

dtStart = dt.datetime.now()

lstRd = list()

#
# Setup RunDesc
#

lstlstNeuron = [[30], [100], [100, 20], [100, 50], [100, 100], [100, 100, 100]]
lstEpochs = [1000, 2000, 5000, 10000]
lstMiniBatches = [100, 200, 500]
# lstlstNeuron = [[30]]
# lstEpochs = [100]
# lstMiniBatches = [50]
for lstNeuronPL, cEpochs, citemsMiniBatch in itertools.product(lstlstNeuron, lstEpochs, lstMiniBatches):
    rd = RunDesc()
    rd.dictArgs = DictArgsDefCreate()
    rd.dictArgs["fUseFullTestSet"] = True
    rd.dictArgs["cEpochs"] = cEpochs
    rd.dictArgs["citemsMiniBatch"] = citemsMiniBatch
    rd.dictArgs["lstcNeuronsPL"] = lstNeuronPL

    lstRd.append(rd)

# Setup output file for logging while training
strOutFile = "/tmp/mnistTrain.csv"
fhLog = open(strOutFile, "w")
strHeader = "CmdLine, RunName, Iter, Accuracy, Dmsec train, lstcNeuronsPL, cEpochs, miniBatch, l2, fUseFullTestSet\n"
fhLog.write(strHeader)

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
        print("[{:0.2f}%] [{}] {} / {}; iter {} / {} - {}".format(
                    100 * float(irun*cIters + iterT + 1) / (len(lstRd) * cIters),
                    StrElapsedTimeFromTd(dt.datetime.now() - dtStart),
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

        # Log results
        strOut = StrResultFromRrRd(rr, rd)
        fhLog.write(strOut)
        fhLog.flush()
    irun = irun + 1

fhLog.close()


print("")
print("Dumped results to CSV '{}'".format(strOutFile))

print("")
print("Total time:  {}".format(StrElapsedTimeFromTd(dt.datetime.now() - dtStart)))
