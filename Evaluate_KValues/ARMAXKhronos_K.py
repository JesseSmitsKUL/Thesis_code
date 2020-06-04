from baseStream import BaseStream
import matplotlib.pyplot as plt
import os
from math import sqrt
import numpy as np
from copy import deepcopy
from Evaluate_KValues.SimARMAXKhronos import SimARMAXKhronos
from collections import defaultdict
from Alternating.createBinaryNN import transformAltStream, createAltModelLSTM, createSVM


def keysValidated(keys, dicKeys):
    for k in keys:
        if dicKeys[k] == -1:
            return False
    return True

def keysSetConstraint(dickeys,keys,minval,k):

    for key in keys:
        if minval > key:
            if dickeys[key] == -1:
                dickeys[key] = k


# def keySmoother(key):
#     from math import exp
#     return exp(1-(1/key**2))
#
# def keySmoother2(key):
#     from math import exp
#     return exp(1-(1/key**3))
#
# def reject_outliers(data, m=6.):
#     d = np.abs(data - np.median(data))
#     mdev = np.median(d)
#     s = d / (mdev if mdev else 1.)
#     return data[s < m]

from statsmodels.tsa.ar_model import AR
class ARMAXKhronos_K(BaseStream):

    def __init__(self, name, times, window, loc):
        super(ARMAXKhronos_K, self).__init__(name, times)
        self.alpha = 7.0 / 8.0
        self.beta = 3.0 / 4.0
        self.variance = -1
        self.constraints_to_K = {0.1: 0, 0.2: 0.1, 0.3: 0.6, 0.4: 1, 0.5: 1.2, 0.6: 1.4, 0.7: 2, 0.8: 2.8, 0.9: 4.9,
                                 0.95: 10.0, 0.99: 280, 1: 300}
        self.meanPE = []
        self.MER = []
        self.variancePE = []
        self.stdPE = []
        self.rmsPE = []
        self.medianPE = []
        self.saveLocation = loc

        self.onlineNN = True
        self.dimNN = 10

        self.window = 300
        self.model = -1
        self.modelAlt = -1
        self.index = -1
        self.windowArrival = window + 2
        self.trendPredictions = []
        self.sameTrendPredictions = []
        self.NNTrendPredictions = []
        self.onTime = []
        self.trendAccuracy = 0
        self.NNTrendAcc = 0
        self.sameTrendAccuracy = 0
        self.modelAlt = 0
        self.altPred = 0
        self.makePlots = True
        self.dump = None
        self.predictionAR = 0
        self.constraints_to_K_unmodified = deepcopy(self.constraints_to_K)
        self.scenario = []
        self.cnt = [-1] * len(times)
        self.sname = ''

        self.comp_window_size = 1000
        self.variances = []

        self.indexToEval = 1200

        self.graph = None


        #self.setupModel()

    def setScenario(self, cnt, scen):
        self.cnt = cnt
        self.scenario = scen


    def setKeys(self, keys):
        self.constrains = keys
        self.constrains.sort()


    def analyseState(self, key):

        if len(self.accuracies_constraints[key]) < self.comp_window_size:
            return 1

        curCompleteness = self.moving_average_accuracy_constraints[key][-1]
        margin = curCompleteness - key
        ratio = (key - 0.001) / (curCompleteness + 0.001)
        if margin <= 0.05:
            # max packets to burn
            maxMissedPackets = round((1 - key) * self.comp_window_size)
            # late is 0 on time is 1
            missedPackets = self.comp_window_size - sum(self.accuracies_constraints[key])
            missmargin = (0.05 * self.comp_window_size) if key <= 0.95 else ((1 - key) * self.comp_window_size)
            canMiss = maxMissedPackets - missedPackets - 1
            if canMiss <= 0:
                print("forced REEVAL for key: ", key)
                print(self.constraints_to_K)
                #self.increaseKey(key)
                print(self.constraints_to_K)
                return 2.5
            else:
                #print(canMiss, " canMiss for key: ", key, " resulting in: ", 1 + 1 *((missmargin - canMiss) / (missmargin)))
                return 1 + 1 *((missmargin-canMiss) / (missmargin))
        else:
            return ratio

    def increaseKey(self,key):
        index = self.constrains.index(key)
        nextKey = self.constrains[index+1]
        print("Mapping key ", key , " to ", nextKey)
        self.constraints_to_K[key] = self.constraints_to_K_unmodified[nextKey]


    # updates SAT
    def update(self):

        dif = self.prediction - self.last_arrival_time
        self.variance = round(self.beta * self.variance + (1 - self.beta) * abs(dif), 2)

        from statsmodels.tsa.ar_model import AR

        begin = max(0, self.index - self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = AR(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.predictionAR = model_fit.predict(len(data), len(data))[0]
        self.prediction = deepcopy(self.predictionAR)

        minVal = 0
        maxVal = 0
        if len(self.predictedArrivals) < 4:
            minVal = min(self.predictedArrivals[-self.windowArrival:])
            maxVal = max(self.predictedArrivals[-self.windowArrival:])
        else:
            cut = self.predictedArrivals[-self.windowArrival:]
            cut.sort()
            minVal = cut[1]
            maxVal = cut[-2]

        if self.prediction < minVal:
            self.prediction = minVal
        elif self.prediction > maxVal:
            self.prediction = maxVal

        previousArrival = -100
        if len(self.predictedArrivals) > 1:
            previousArrival = self.predictedArrivals[-1]



        for key in self.constrains:

            new_timeout = round(self.prediction + (self.constraints_to_K[key]) * self.analyseState(key) * self.variance,3)
            if previousArrival != -100:
                predTrend = 1 if self.predictionAR - previousArrival > 0 else 0

                if predTrend == int(self.altPred) == 0:
                    new_timeout = round(self.prediction + (self.constraints_to_K[key]) * self.analyseState(key) / 1.5 * self.variance, 3)

            # limit = max(self.predictedArrivals[-self.window:]) + abs(min(self.resultDifferences[-self.window:]))
            # new_timeout = min(limit,new_timeout)
            self.timeouts[key].append(new_timeout)


    def initialConfiguration(self, lat):
        self.prediction = lat
        self.variance = (max(self.arrivals[:self.index]) - min(self.arrivals[:self.index]))
        self.last_arrival_time = lat
        self.timeouts = dict()
        self.onTime = dict()

        for key in self.constrains:
            self.timeouts[key] = [lat]
            self.accuracies_constraints[key] = []
            self.moving_average_accuracy_constraints[key] = []
            self.moving_average_achieved_completeness[key] = []
            self.moving_average_achieved_completenesses[key] = []
            self.constraint_violations[key] = []
            self.timeoutDifferences[key] = []
            self.onTime[key] = []

        self.meanSquaredError = []
        self.meanRootSquaredError = []
        self.meanAbsError = []
        self.stdAbsError = []

    def setupModel(self):
        data = max(self.comp_window_size, self.index)
        start = data - 3000 if data >= 3000 else 0
        x_set, y_set = transformAltStream(self.arrivals[start:data], 10)
        streamModel, _ = createSVM(x_set, y_set)
        self.modelAlt = streamModel
        return

        # if self.onlineNN:
        #     data = max(self.comp_window_size*2, self.index)
        #     start = data-5000 if data >= 5000 else 0
        #     x_set, y_set = transformAltStream(self.arrivals[start:data], self.dimNN)
        #     streamModel, hist = createAltModelLSTM(x_set, y_set)
        #     streamModel.summary()
        #     self.modelAlt = streamModel
        # else:
        #     self.dimNN = 5
        #     file = '../Alternating/alt_ref_model.h5'
        #     from keras.models import load_model
        #     import tensorflow
        #     self.modelAlt = tensorflow.keras.models.load_model(file)


    def predictAltenating(self):
        if self.index < self.dimNN:
            return 1

        X = [self.arrivals[self.index-self.dimNN:self.index]]
        X = np.array(X)
        #X = X.reshape((X.shape[0], X.shape[1], 1))

        pred = self.modelAlt.predict(X)[0]
        #print("pred", pred)
        self.altPred = round(pred)

    def updateKVal(self):

        newConstraints = {}
        newConstraints = defaultdict(lambda: -1, newConstraints)


        k_values = [x * 0.1 for x in range(0, 10000)]
        for k in k_values:
            Keval = SimARMAXKhronos("",self.predictedArrivals,self.windowArrival,False,self.results,self.variances,k)
            Keval.runSimulation()
            minConstraint = min(Keval.moving_average_accuracy_constraints)
            keysSetConstraint(newConstraints,self.constrains, minConstraint, k)

            if keysValidated(self.constrains, newConstraints):
                for key in self.constrains:
                    self.constraints_to_K[key] = newConstraints[key] + key

                self.constraints_to_K_unmodified = deepcopy(self.constraints_to_K)
                print(self.constraints_to_K)
                return


    def increment(self, rat):
        if len(self.variances) >= (self.indexToEval-2) and len(self.variances) % 200 == 0:
            self.updateKVal()
        if self.index % 200 == 0:
            self.setupModel()

        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.predictAltenating()
            self.update()

    def updateArrival(self):
        if self.index == -1:
            self.setupModel()


            for x in range(1):
                packet = self.arrivals[x]
                # self.increment(packet)
                self.index = 10
                self.initialConfiguration(packet)
        else:
            packet = self.arrivals[self.index]


            previousArrival = -100
            if len(self.predictedArrivals) > 1:
                previousArrival = self.predictedArrivals[-1]

            self.predictedArrivals.append(packet)
            self.results.append(self.prediction)
            self.variances.append(self.variance)
            self.resultDifferences.append(self.prediction - packet)

            self.predictAltenating()

            if previousArrival != -100:

                actualTrend = 1 if packet - previousArrival > 0 else 0
                predTrend = 1 if self.predictionAR - previousArrival > 0 else 0

                # print("Prev: ", previousArrival, " Curr: ", packet, " Actual-Pred trand ", actualTrend, " , ", predTrend )

                if actualTrend != int(self.altPred):  # false predictions
                    self.NNTrendPredictions.append(0)
                else:  # correct prediction
                    self.NNTrendPredictions.append(1)

                if actualTrend != predTrend:  # false predictions
                    self.trendPredictions.append(0)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(0)
                else:  # correct prediction
                    self.trendPredictions.append(1)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(1)


            for key in self.constrains:
                timeout = self.timeouts[key][-1]
                ontime = 1 if packet < timeout else 0
                self.onTime[key].append(ontime)

                if len(self.accuracies_constraints[key]) < self.comp_window_size:
                    self.accuracies_constraints[key].append(ontime)
                else:
                    self.accuracies_constraints[key] = self.accuracies_constraints[key][1:] + [ontime]
                # print('acc_cons: ', self.accuracies_constraints[key])

                self.timeoutDifferences[key].append(round(timeout - packet, 2))

            self.updateMovingAverageAccCons()
            self.updateCompleteness()
            self.updateViolation()
            self.increment(packet)

    def runSimulation(self):

        self.setupModel()

        for x in range(1):
            packet = self.arrivals[x]
            # self.increment(packet)
            self.index = 10
            self.initialConfiguration(packet)

        #print("cnt")
        #print(self.cnt)
        lastCnt = self.cnt[9]
        for (packet,cnt) in zip(self.arrivals[10:],self.cnt[10:]):

            if cnt != -1 and lastCnt + 1 != cnt:
                print("OUT OF ORDER: ", packet)
                lastCnt = cnt
                continue

            lastCnt = cnt


            previousArrival = -100
            if len(self.predictedArrivals) > 1:
                previousArrival = self.predictedArrivals[-1]

            self.predictedArrivals.append(packet)
            self.results.append(self.prediction)
            self.variances.append(self.variance)
            self.resultDifferences.append(self.prediction - packet)

            self.predictAltenating()

            if self.index > self.indexToEval + 2:

                actualTrend = 1 if packet - previousArrival > 0 else 0
                predTrend = 1 if self.predictionAR - previousArrival > 0 else 0

                # print("Prev: ", previousArrival, " Curr: ", packet, " Actual-Pred trand ", actualTrend, " , ", predTrend )

                if actualTrend != int(self.altPred):  # false predictions
                    self.NNTrendPredictions.append(0)
                else:  # correct prediction
                    self.NNTrendPredictions.append(1)

                if actualTrend != predTrend:  # false predictions
                    self.trendPredictions.append(0)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(0)
                else:  # correct prediction
                    self.trendPredictions.append(1)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(1)


            for key in self.constrains:
                timeout = self.timeouts[key][-1]
                ontime = 1 if packet < timeout else 0
                self.onTime[key].append(ontime)

                if len(self.accuracies_constraints[key]) < self.comp_window_size:
                    self.accuracies_constraints[key].append(ontime)
                else:
                    self.accuracies_constraints[key] = self.accuracies_constraints[key][1:] + [ontime]
                # print('acc_cons: ', self.accuracies_constraints[key])

                self.timeoutDifferences[key].append(round(timeout - packet, 2))

            self.updateMovingAverageAccCons()
            self.updateCompleteness()
            self.updateViolation()
            self.increment(packet)

        self.calculateMetrics()

        if self.filewriter != None:
            self.filewriter.write('\nMAE: '+ str(self.meanAbsError))
            self.filewriter.write("\nTrendAcc: " + str(self.trendAccuracy) + '\t\t Altmodel: ' + str(self.NNTrendAcc) +
                  '\t\t Same pred ACC: ' + str(self.sameTrendAccuracy) + ', for ' + str(len(self.sameTrendPredictions)))
            self.filewriter.write('\nMissedEventRatio: ' + str(self.MER))
            keyvio = [(key, sum(self.constraint_violations[key][self.indexToEval*2:])) for key in self.constrains]
            self.filewriter.write("\nVIOLATIONS: " + str(keyvio) + "\n")
            for p in keyvio:
                if p[1] != 0:
                    self.dump.write("\nFAIL: " + "\n")

        print('MAE: ', self.meanAbsError) #, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ",self.meanRootSquaredError\
        print("TrendAcc: ", self.trendAccuracy, '\t\t Altmodel: ', self.NNTrendAcc,
              '\t\t Same pred ACC: ', self.sameTrendAccuracy, ', for ', len(self.sameTrendPredictions))
        print('MissedEventRatio: ' + str(self.MER))
        keyvio = []
        for key in self.constrains:
            if len(self.arrivals) < self.indexToEval:
                keyvio.append((key,self.constraint_violations[key][-1]))
            else:
                keyvio.append((key,sum(self.constraint_violations[key][self.indexToEval:])))

        for p in keyvio:
            if p[1]!= 0:
                print("FAILED CONSTRAINT")

        print("VIOLATIONS ######### ", keyvio)

        if self.makePlots:
            self.visualizeData()

    def updateMovingAverageAccCons(self):
        for key in self.constrains:
            length = len(self.accuracies_constraints[key])
            average = sum(self.accuracies_constraints[key])
            moving_av = round(average / length, 2)

            if len(self.moving_average_accuracy_constraints[key]) < self.comp_window_size:
                self.moving_average_accuracy_constraints[key].append(moving_av)
            else:
                self.moving_average_accuracy_constraints[key] = self.moving_average_accuracy_constraints[key][1:] + [
                    moving_av]
            # print('updateMovinceAvAccCons:', self.moving_average_accuracy_constraints[key])

    def updateCompleteness(self):
        for key in self.constrains:
            count_above_completeness = sum(mov_av >= key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage, 2)

            if len(self.moving_average_achieved_completeness[key]) < self.comp_window_size:
                self.moving_average_achieved_completeness[key].append(percentage)
            else:
                self.moving_average_achieved_completeness[key] = self.moving_average_achieved_completeness[key][1:] + [
                    percentage]
            # print('UpdateCompleteness: ',  self.moving_average_achieved_completeness[key])

    def updateViolation(self):
        for key in self.constrains:
            count_above_completeness = sum(
                mov_av < key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage, 2)
            self.constraint_violations[key].append(percentage)
        # print('UpdateViolation: ', self.constraint_violations[keyv])

    def calculateMetrics(self):

        for key in self.constrains:
            self.meanAbsError.append(sum(map(abs, self.timeoutDifferences[key][self.indexToEval:])) / len(self.timeoutDifferences[key][self.indexToEval:]))
            self.meanSquaredError.append(
                sum(map(lambda x: x * x, self.timeoutDifferences[key][self.indexToEval:])) / len(self.timeoutDifferences[key][self.indexToEval:]))
            self.meanRootSquaredError.append(
                sqrt(sum(map(lambda x: x * x, self.timeoutDifferences[key][self.indexToEval:])) / len(self.timeoutDifferences[key][self.indexToEval:])))
            self.MER.append(sum(self.onTime[key][self.indexToEval:]) / len(self.onTime[key][self.indexToEval:]))
        self.trendAccuracy = sum(self.trendPredictions) / len(self.trendPredictions)
        self.sameTrendAccuracy = sum(self.sameTrendPredictions) / len(self.sameTrendPredictions)
        self.NNTrendAcc = sum(self.NNTrendPredictions) / len(self.NNTrendPredictions)


    def visualizeData(self):

        if self.makePlots == False:
            return


        plt.plot(self.arrivals, label='arrival times')
        plt.ylabel('arrival time (seconds)')
        plt.title('Arrival times in CPS')

        if self.savePlots:
            plt.savefig(self.saveLocation + "arrivaltimes.png")
            plt.close()
        else:
            plt.show()

        for key in self.constrains:


            if self.savePlots:
                keyroot = self.saveLocation + str(key) + '/'
                directory = os.path.dirname(keyroot)
                if not os.path.exists(directory):
                    os.makedirs(directory)


            plt.plot(self.constraint_violations[key], color='red')
            plt.ylabel('Constraint violation (%) '+ str(key) + " " + self.name)

            if self.savePlots:
                plt.savefig(keyroot + "violations" + ".png")
                plt.close()
            else:
                plt.show()



            datapoints = 200
            for x in range(1,15):
                # plt.plot(self.predictedArrivals[1000 + (x - 1) * datapoints:1000 + x * datapoints], label='arrival')
                # plt.plot(self.timeouts[key][1000 + (x - 1) * datapoints:1000 + x * datapoints], label='prediction')
                plt.plot(self.predictedArrivals[(x - 1) * datapoints: x * datapoints], label='arrival')
                plt.plot(self.timeouts[key][(x - 1) * datapoints:  x * datapoints], label='timeout')
                plt.legend()
                plt.ylabel(str(key) + "  " + self.name[-4:])

                if self.savePlots:
                    plt.savefig(keyroot + "predictions" + str(x) + ".png")
                    plt.close()
                else:
                    plt.show()

            plt.plot(self.predictedArrivals, label='arrival')
            plt.plot(self.timeouts[key], label='prediction')
            plt.plot(self.results, label='res')
            plt.legend()
            plt.ylabel('All ' + str(key) + "  " + self.name[-4:])
            if self.savePlots:
                plt.savefig(keyroot + "allpredictions.png")
                plt.close()
            else:
                plt.show()
