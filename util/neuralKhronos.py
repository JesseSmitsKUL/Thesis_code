from baseStream import BaseStream
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.ar_model import AR
from math import sqrt
import math
from createNN import *
import numpy as np


# file used to make Khronos with NN arrival time predictions


def keySmoother(key):
    from math import exp
    return exp(1-(1/key**2))

def keySmoother2(key):
    from math import exp
    return exp(1-(1/key**3))

def reject_outliers_2(data, m=3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]



class NNKhronos(BaseStream):

    def __init__(self,name,times, model, cutoff, save):
        super(NNKhronos, self).__init__(name, times)
        self.alpha = 7.0/8.0
        self.beta = 3.0/4
        self.variance = -1
        self.constraints_to_K = {0.1: 0, 0.2: 0.1, 0.3: 0.6, 0.4: 1, 0.5: 1.2, 0.6: 1.4, 0.7: 2, 0.8: 2.8, 0.9: 4.9,
                                 0.95: 10, 0.99: 280, 1: 300}
        self.modelName = model
        self.cutoff = cutoff
        self.LEN_STREAM = 30
        self.nn_window = []
        self.model = None
        self.meanPE = []
        self.variancePE = []
        self.stdPE = []
        self.rmsPE =[]
        self.medianPE = []
        self.save = save
        self.setupModel()

    def setKeys(self, keys):
        self.constrains = keys

    def analyseState(self,key):

        if len(self.accuracies_constraints[key]) < self.comp_window_size:
            return 1

        curCompleteness = self.moving_average_accuracy_constraints[key][-1]
        factor = curCompleteness - key
        ratio = (key-0.1) / (curCompleteness + 0.001)
        if factor < 0.05:
            # going from -x -> 0.05
            factor = 0.05 - max(0,factor) # go for 1 to 3 depending on packets left the burn out
            # max packets to burn
            maxMissedPackets = round((1-key)*self.comp_window_size)
            # late is 0 on time is 1
            missedPackets = self.comp_window_size - sum(self.accuracies_constraints[key])
            canMiss = maxMissedPackets - missedPackets
            if canMiss <= 0:
                return 2.0
            else:
                return 1+1.0*(missedPackets/maxMissedPackets)
        else:
            return ratio

    def analyseState2(self,key,interset):
        curCompleteness = self.moving_average_accuracy_constraints[key][-1]
        factor = curCompleteness - key
        ratio = key / (curCompleteness + 0.001)
        if factor < 0.05:
            toViolationPercentage = (1-factor)
            maxEle = np.amax(interset)*toViolationPercentage
            return maxEle/np.mean(interset)
        else:
            return ratio


    # updates SAT
    def update(self):
        series = np.array([self.nn_window])
        if 'cnn' in self.modelName:
            series = series.reshape((series.shape[0], series.shape[1], 1))

        self.variance = round(self.beta * self.variance + (1 - self.beta) * abs(self.prediction - self.last_arrival_time), 2)
        self.prediction = round(self.model.predict(series)[0][0], 2)

        interset = self.resultDifferences[-100:] if len(self.resultDifferences) > 100 else self.resultDifferences
        intersetWO = np.abs(array(interset))
        interset = reject_outliers_2(intersetWO)

        self.meanPE.append(np.mean(interset))
        self.variancePE.append(interset.var())
        self.stdPE.append(np.std(interset))
        self.medianPE.append(np.median(interset))
        self.rmsPE.append(math.sqrt(np.square(interset).mean()))

        for key in self.constrains:
            new_timeout = round(self.prediction + self.constraints_to_K[key] * self.variance,3)
            self.timeouts[key].append(new_timeout)
        return

        for key in self.constrains:
            index = math.ceil(len(interset) * key)
            index = min(index,len(interset)-1)
            interset.sort()
            element = interset[index]
            margin = (keySmoother2(key) * 2 * np.median(interset)) + element


            # new_timeout = round(self.prediction + self.constraints_to_K[key] * self.variance, 3)
            new_timeout = round(self.prediction + (min(margin,np.amax(intersetWO)))*self.analyseState(key), 3)
            # new_timeout = round(self.prediction + np.mean(intersetWO) + keySmoother(key) * 2 * math.sqrt(np.square(intersetWO).mean()) , 3)
            # new_timeout = round(self.prediction + (np.mean(interset) * self.analyseState(key)) + keySmoother(key) * 2 * math.sqrt(np.square(interset).mean()) , 3)
            # new_timeout = round(self.prediction + np.mean(interset) + keySmoother(key) * np.square(interset).mean() , 3)

            
            self.timeouts[key].append(new_timeout)



        # for key in self.constrains:
        #     set = self.resultDifferences[-300:] if len(self.resultDifferences) > 100 else self.resultDifferences
        #     set = [abs(x) for x in set]
        #     set.sort()
        #     index = math.floor(len(set)*key)
        #     print(index, len(set))
        #     new_timeout = round(self.prediction + set[index],3)
        #     self.timeouts[key].append(new_timeout)

        #print('timeouts:' , self.timeouts)



    def initialConfiguration(self,lat):
        self.prediction = lat
        self.variance = float(lat)/2
        self.last_arrival_time = lat
        self.timeouts = dict()
        for key in self.constrains:
            self.timeouts[key] = [lat]
            self.accuracies_constraints[key] = []
            self.moving_average_accuracy_constraints[key] = []
            self.moving_average_achieved_completeness[key] = []
            self.moving_average_achieved_completenesses[key] = []
            self.constraint_violations[key] = []
            self.timeoutDifferences[key] = []
            self.meanSquaredError = []
            self.meanRootSquaredError = []
            self.meanAbsError = []
            self.timeoutDifferences[key] = []


    def setupModel(self):

        saveFile = './ref_' + self.save + '.h5'

        import os
        from keras.models import load_model
        exists = os.path.isfile(saveFile)

        modelname = self.modelName
        cutoff = round(len(self.arrivals) * self.cutoff)
        x_train, y_train = transformStream(self.arrivals[:cutoff], self.LEN_STREAM)
        self.arrivals = self.arrivals[cutoff:]


        # if exists:
        #     self.model = load_model(saveFile)
        #     return

        if modelname == 'cnn':
            self.model, history = createModelCNN(x_train, y_train)
        elif modelname == 'cnn2':
            self.model, history = createModelCNN2(x_train, y_train)
        elif modelname == 'gru':
            self.model, history = createModelGRU(x_train, y_train)
        elif modelname == 'gru2':
            self.model, history = createModelGRU2(x_train, y_train)
        elif modelname == 'cnn_lstm':
            self.model, history = createModelCNN_LSTM(x_train, y_train)
        elif modelname == 'cnn_lstm2':
            self.model, history = createModelCNN_LSTM2(x_train, y_train)
        elif modelname == 'cnn_gru':
            self.model, history = createModelCNN_GRU(x_train, y_train)
        elif modelname == 'cnn_gru2':
            self.model, history = createModelCNN_GRU2(x_train, y_train)
        elif modelname == 'cnn_gru_test':
            self.model, history = createModelCNN_GRU_TEST(x_train, y_train)
        else:
            print("Cannot run the given model: ", self.model)

        self.model.save(saveFile)

    def increment(self,rat):

        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.nn_window = self.nn_window[1:] + [rat]
            self.last_arrival_time = rat
            self.update()

    def runSimulation(self):


        for packet in self.arrivals:

            if len(self.nn_window) < self.LEN_STREAM-1:
                self.nn_window.append(packet)
            elif len(self.nn_window) == self.LEN_STREAM-1:
                self.nn_window.append(packet)
                self.increment(packet)
            else:
                for key in self.constrains:
                    timeout = self.timeouts[key][-1]
                    ontime = 1 if packet < timeout else 0

                    if len(self.accuracies_constraints[key]) < self.comp_window_size:
                        self.accuracies_constraints[key].append(ontime)
                    else:
                        self.accuracies_constraints[key] = self.accuracies_constraints[key][1:] + [ontime]
                    # print('acc_cons: ', self.accuracies_constraints[key])

                    self.resultDifferences.append(round(self.prediction-packet,2))
                    self.timeoutDifferences[key].append(round(timeout-packet,2))

                self.predictedArrivals.append(packet)
                self.updateMovingAverageAccCons()
                self.updateCompleteness()
                self.updateViolation()
                self.increment(packet)

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError)


    def updateMovingAverageAccCons(self):
        for key in self.constrains:
            length = len(self.accuracies_constraints[key])
            average = sum(self.accuracies_constraints[key])
            moving_av = round(average/length,2)

            if len(self.moving_average_accuracy_constraints[key]) < self.comp_window_size:
                self.moving_average_accuracy_constraints[key].append(moving_av)
            else:
                self.moving_average_accuracy_constraints[key] = self.moving_average_accuracy_constraints[key][1:] + [moving_av]
            # print('updateMovinceAvAccCons:', self.moving_average_accuracy_constraints[key])

    def updateCompleteness(self):
        for key in self.constrains:
            count_above_completeness = sum(mov_av >= key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness*100)/len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage,2)

            if len(self.moving_average_achieved_completeness[key]) < self.comp_window_size:
                self.moving_average_achieved_completeness[key].append(percentage)
            else:
                self.moving_average_achieved_completeness[key] = self.moving_average_achieved_completeness[key][1:] + [percentage]
            # print('UpdateCompleteness: ',  self.moving_average_achieved_completeness[key])

    def updateViolation(self):
        # for key in self.constrains:
            # count_below_completeness = sum(completeness < key*100 for completeness in self.moving_average_achieved_completeness[key])
            # percentage = (count_below_completeness*100)/len(self.moving_average_achieved_completeness[key])
            # self.constraint_violations[key].append(percentage)
            # print('UpdateViolation: ', self.constraint_violations[key])
        keyv = 0
        for key in self.constrains:
            keyv = key
            count_above_completeness = sum(
                mov_av < key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage, 2)
            self.constraint_violations[key].append(percentage)
        print('UpdateViolation: ', self.constraint_violations[keyv])





    def calculateMetrics(self):

        for key in self.constrains:

            self.meanAbsError.append(sum(map(abs,self.timeoutDifferences[key]))/len(self.timeoutDifferences[key]))
            self.meanSquaredError.append(sum(map(lambda x:x*x,self.timeoutDifferences[key]))/len(self.timeoutDifferences[key]))
            self.meanRootSquaredError.append(sqrt(sum(map(lambda x:x*x,self.timeoutDifferences[key]))/len(self.timeoutDifferences[key])))


    def visualizeData(self):

        if self.makePlots == False:
            return

        if self.savePlots:
            directory = os.path.dirname(self.saveLocation)
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.plot(self.arrivals, label='arrival times')
        plt.ylabel('arrival time (seconds)')
        plt.title('Arrival times in CPS')
        plt.show()


        datapoints = 350
        for key in self.constrains:
            for x in range(3):
                plt.plot(self.predictedArrivals[400+(x-1)*datapoints:400+x*datapoints], label='arrival')
                plt.plot(self.timeouts[key][400+(x-1)*datapoints:400+x*datapoints], label='prediction')
                plt.legend()
                plt.ylabel(str(key) + "  " + self.name[-4:])

                if self.savePlots:
                    plt.savefig(directory + self.prefix + self.name + "_Predictions" + ".png")

                plt.show()

        fig, ax = plt.subplots()
        ax.boxplot(self.timeoutDifferences.values())
        ax.set_xticklabels(self.timeoutDifferences.keys())
        fig.show()
        #
        # for key in self.constrains:
        #     fig4, ax4 = plt.subplots()
        #     ax4.set_title('Hide Outlier Points')
        #     ax4.boxplot(self.timeoutDifferences[key], showfliers=False)
        #
        #     if self.savePlots:
        #         fig4.savefig(directory + self.prefix + self.name + "_Boxplot_Error" + ".png")
        #
        #     fig4.show()
        #
        #     absDif = np.absolute(self.resultDifferences)
        #     absDif[absDif > 20] = 25
        #
        #     plt.plot(self.stdPE, label='std')
        #     # plt.plot(self.variancePE, label='variance')
        #     plt.plot(self.meanPE, label='mean')
        #     plt.plot(self.rmsPE, label='root MSE')
        #     plt.plot(self.medianPE, label='median er')
        #
        #     plt.legend()
        #     plt.show()
        #
        #     plt.plot(absDif)
        #     plt.show()


        for key in self.constrains:
            plt.plot(self.timeoutDifferences[key][100:600], color='red')
            plt.axhline(y=0, color='green', linestyle='-')
            plt.ylabel('differences' + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()

        for key in self.constrains:
            plt.plot(self.constraint_violations[key], color='red')
            plt.ylabel('Constraint violation (\%) ' + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()


from statsmodels.tsa.ar_model import AR
class ARMAXKhronos(BaseStream):

    def __init__(self, name, times, window, save):
        super(ARMAXKhronos, self).__init__(name, times)
        self.alpha = 7.0 / 8.0
        self.beta = 3.0 / 4
        self.variance = -1
        self.constraints_to_K = {0.1: 0, 0.2: 0.1, 0.3: 0.6, 0.4: 1, 0.5: 1.2, 0.6: 1.4, 0.7: 2, 0.8: 2.8, 0.9: 4.9,
                                 0.95: 10, 0.99: 280, 1: 300}
        self.meanPE = []
        self.variancePE = []
        self.stdPE = []
        self.rmsPE = []
        self.medianPE = []
        self.save = save

        self.window = 500
        self.model = -1
        self.modelAlt = -1
        self.index = -1
        self.windowArrival = window + 2
        self.trendPredictions = []
        self.sameTrendPredictions = []
        self.NNTrendPredictions = []
        self.trendAccuracy = 0
        self.NNTrendAcc = 0
        self.sameTrendAccuracy = 0
        self.modelAlt = 0
        self.altPred = 0
        self.makePlots = False


        self.setupModel()

    def setKeys(self, keys):
        self.constrains = keys

    def updateKeys(self):

        # keep arrival times and predictions
        # check timeout errors

        pass

    def analyseState(self, key):

        if len(self.accuracies_constraints[key]) < self.comp_window_size:
            return 1

        curCompleteness = self.moving_average_accuracy_constraints[key][-1]
        factor = curCompleteness - key
        ratio = (key - 0.1) / (curCompleteness + 0.001)
        if factor < 0.05:
            # going from -x -> 0.05
            factor = 0.05 - max(0, factor)  # go for 1 to 3 depending on packets left the burn out
            # max packets to burn
            maxMissedPackets = round((1 - key) * self.comp_window_size)
            # late is 0 on time is 1
            missedPackets = self.comp_window_size - sum(self.accuracies_constraints[key])
            canMiss = maxMissedPackets - missedPackets
            if canMiss <= 0:
                return 2.0
            else:
                return 1 + 1.0 * (missedPackets / maxMissedPackets)
        else:
            return ratio

    def analyseState2(self, key, interset):
        curCompleteness = self.moving_average_accuracy_constraints[key][-1]
        factor = curCompleteness - key
        ratio = key / (curCompleteness + 0.001)
        if factor < 0.05:
            toViolationPercentage = (1 - factor)
            maxEle = np.amax(interset) * toViolationPercentage
            return maxEle / np.mean(interset)
        else:
            return ratio

    # updates SAT
    def update(self):


        self.variance = round(self.beta * self.variance + (1 - self.beta) * abs(self.prediction - self.last_arrival_time), 2)

        from statsmodels.tsa.ar_model import AR

        begin = max(0, self.index - self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = AR(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]

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

        interset = self.resultDifferences[-300:] if len(self.resultDifferences) > 300 else self.resultDifferences
        intersetWO = np.abs(array(interset))
        interset = reject_outliers_2(intersetWO)



        self.meanPE.append(np.mean(interset))
        self.variancePE.append(interset.var())
        self.stdPE.append(np.std(interset))
        self.medianPE.append(np.median(interset))
        self.rmsPE.append(math.sqrt(np.square(interset).mean()))

        interset = self.resultDifferences[-200:] if len(self.resultDifferences) > 200 else self.resultDifferences

        previousArrival = -100
        if len(self.predictedArrivals) > 1:
            previousArrival = self.predictedArrivals[-1]

        for key in self.constrains:

            new_timeout = new_timeout = round(self.prediction + (self.constraints_to_K[key]) * self.variance,3)
            if previousArrival != -100:
                predTrend = 1 if self.prediction - previousArrival > 0 else 0

                if predTrend == int(self.altPred):
                    new_timeout = round(self.prediction + (self.constraints_to_K[key]) * self.variance, 3)

            self.timeouts[key].append(new_timeout)
            continue


            index = math.ceil(len(interset) * key)
            index = min(index, len(interset) - 1)
            #interset = [abs(x) for x in interset]
            interset.sort()
            element = interset[index]
            margin = (keySmoother2(key)* 2 * np.median(interset)) + element
            extreme = interset[-2] if len(interset) > 1 else interset[-1]
            new_timeout = round(self.prediction + extreme, 3)
            # new_timeout = round(self.prediction + self.constraints_to_K[key] * self.variance, 3)
            #new_timeout = round(self.prediction + (min(margin, np.amax(intersetWO))) * self.analyseState(key), 3)
            # new_timeout = round(self.prediction + np.mean(intersetWO) + keySmoother(key) * 2 * math.sqrt(np.square(intersetWO).mean()) , 3)
            # new_timeout = round(self.prediction + (np.mean(interset) * self.analyseState(key)) + keySmoother(key) * 2 * math.sqrt(np.square(interset).mean()) , 3)
            # new_timeout = round(self.prediction + np.mean(interset) + keySmoother(key) * np.square(interset).mean() , 3)

            # self.timeouts[key].append(new_timeout)

        # for key in self.constrains:
        #     set = self.resultDifferences[-300:] if len(self.resultDifferences) > 100 else self.resultDifferences
        #     set = [abs(x) for x in set]
        #     set.sort()
        #     index = math.floor(len(set)*key)
        #     print(index, len(set))
        #     new_timeout = round(self.prediction + set[index],3)
        #     self.timeouts[key].append(new_timeout)

        # print('timeouts:' , self.timeouts)

    def initialConfiguration(self, lat):
        self.prediction = lat
        self.variance = (max(self.arrivals[:self.index]) - min(self.arrivals[:self.index]))
        self.last_arrival_time = lat
        self.timeouts = dict()

        for key in self.constrains:
            self.timeouts[key] = [lat]
            self.accuracies_constraints[key] = []
            self.moving_average_accuracy_constraints[key] = []
            self.moving_average_achieved_completeness[key] = []
            self.moving_average_achieved_completenesses[key] = []
            self.constraint_violations[key] = []
            self.timeoutDifferences[key] = []
            self.meanSquaredError = []
            self.meanRootSquaredError = []
            self.meanAbsError = []
            self.timeoutDifferences[key] = []

    def setupModel(self):

        file = './Alternating/alt_ref_model.h5'

        from keras.models import load_model
        self.modelAlt = load_model(file)

    def predictAltenating(self):

        X = [self.arrivals[self.index-5:self.index]]
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        pred = self.modelAlt.predict(X)[0]
        self.altPred = round(pred[0])

    def increment(self, rat):

        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.predictAltenating()
            self.update()

    def runSimulation(self):

        for x in range(1):
            packet = self.arrivals[x]
            # self.increment(packet)
            self.index = 10
            self.initialConfiguration(packet)

        for packet in self.arrivals[10:]:

            previousArrival = -100
            if len(self.predictedArrivals) > 1:
                previousArrival = self.predictedArrivals[-1]

            self.predictedArrivals.append(packet)
            self.results.append(self.prediction)
            self.resultDifferences.append(self.prediction - packet)

            self.predictAltenating()

            if previousArrival != -100:

                actualTrend = 1 if packet - previousArrival > 0 else 0
                predTrend = 1 if self.prediction - previousArrival > 0 else 0

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

        print('MAE: ', self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ",
              self.meanRootSquaredError, "\nTrendAcc: ", self.trendAccuracy, '\t\t NNACC: ', self.NNTrendAcc,
              '\t\t Same pred ACC: ', self.sameTrendAccuracy, ', for ', len(self.sameTrendPredictions))

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
        # for key in self.constrains:
        # count_below_completeness = sum(completeness < key*100 for completeness in self.moving_average_achieved_completeness[key])
        # percentage = (count_below_completeness*100)/len(self.moving_average_achieved_completeness[key])
        # self.constraint_violations[key].append(percentage)
        # print('UpdateViolation: ', self.constraint_violations[key])
        keyv = 0
        for key in self.constrains:
            keyv = key
            count_above_completeness = sum(
                mov_av < key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage, 2)
            self.constraint_violations[key].append(percentage)
        # print('UpdateViolation: ', self.constraint_violations[keyv])

    def calculateMetrics(self):

        for key in self.constrains:
            self.meanAbsError.append(sum(map(abs, self.timeoutDifferences[key])) / len(self.timeoutDifferences[key]))
            self.meanSquaredError.append(
                sum(map(lambda x: x * x, self.timeoutDifferences[key])) / len(self.timeoutDifferences[key]))
            self.meanRootSquaredError.append(
                sqrt(sum(map(lambda x: x * x, self.timeoutDifferences[key])) / len(self.timeoutDifferences[key])))
        self.trendAccuracy = sum(self.trendPredictions) / len(self.trendPredictions)
        self.sameTrendAccuracy = sum(self.sameTrendPredictions) / len(self.sameTrendPredictions)
        self.NNTrendAcc = sum(self.NNTrendPredictions) / len(self.NNTrendPredictions)


    def visualizeData(self):


        self.makePlots = True
        if self.makePlots == False:
            return

        if self.savePlots:
            directory = os.path.dirname(self.saveLocation)
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.plot(self.arrivals, label='arrival times')
        plt.ylabel('arrival time (seconds)')
        plt.title('Arrival times in CPS')
        plt.show()

        datapoints = 350
        for key in self.constrains:
            for x in range(3):
                plt.plot(self.predictedArrivals[400 + (x - 1) * datapoints:400 + x * datapoints], label='arrival')
                plt.plot(self.timeouts[key][400 + (x - 1) * datapoints:400 + x * datapoints], label='prediction')
                plt.legend()
                plt.ylabel(str(key) + "  " + self.name[-4:])

                if self.savePlots:
                    plt.savefig(directory + self.prefix + self.name + "_Predictions" + ".png")

                plt.show()

            plt.plot(self.predictedArrivals, label='arrival')
            plt.plot(self.timeouts[key], label='prediction')
            plt.legend()
            plt.ylabel('All ' + str(key) + "  " + self.name[-4:])
            plt.show()


        fig, ax = plt.subplots()
        ax.boxplot(self.timeoutDifferences.values())
        ax.set_xticklabels(self.timeoutDifferences.keys())
        fig.show()
        #
        # for key in self.constrains:
        #     fig4, ax4 = plt.subplots()
        #     ax4.set_title('Hide Outlier Points')
        #     ax4.boxplot(self.timeoutDifferences[key], showfliers=False)
        #
        #     if self.savePlots:
        #         fig4.savefig(directory + self.prefix + self.name + "_Boxplot_Error" + ".png")
        #
        #     fig4.show()
        #
        #     absDif = np.absolute(self.resultDifferences)
        #     absDif[absDif > 20] = 25
        #
        #     plt.plot(self.stdPE, label='std')
        #     # plt.plot(self.variancePE, label='variance')
        #     plt.plot(self.meanPE, label='mean')
        #     plt.plot(self.rmsPE, label='root MSE')
        #     plt.plot(self.medianPE, label='median er')
        #
        #     plt.legend()
        #     plt.show()
        #
        #     plt.plot(absDif)
        #     plt.show()

        for key in self.constrains:
            plt.plot(self.timeoutDifferences[key][100:600], color='red')
            plt.axhline(y=0, color='green', linestyle='-')
            plt.ylabel('differences' + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()

        for key in self.constrains:
            plt.plot(self.constraint_violations[key], color='red')
            plt.ylabel('Constraint violation (%) '+ str(key) + " " + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()

