from baseStream import BaseStream
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from random import random
import numpy as np

import matplotlib.pyplot as plt
import os
from math import sqrt


class ARStreamSES(BaseStream):

    def __init__(self,name,times, window):
        super(ARStreamSES, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1
        self.windowArrival = window

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = AR(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]

        minVal = min(self.predictedArrivals[-self.windowArrival:])
        maxVal = max(self.predictedArrivals[-self.windowArrival:])

        if self.prediction < minVal or self.prediction > maxVal:
            model = SimpleExpSmoothing(data)
            model_fit = model.fit()
            self.prediction = model_fit.predict(len(data), len(data))[0]







    def initialConfiguration(self,lat):
        print('Data for AR model with TW: ', self.window)
        self.last_arrival_time = lat
        self.index = 0
        self.prediction = lat

    def increment(self,rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.update()


    def runSimulation(self):

        for x in range(1):
            packet = self.arrivals[x]
            #self.increment(packet)
            self.initialConfiguration(packet)
            self.index = 10

        for packet in self.arrivals[10:]:
            #print('making prediction for: ', packet)
            #print("Packet ", packet)
            self.predictedArrivals.append(packet)
            self.resultDifferences.append(self.prediction - packet)
            self.results.append(self.prediction)
            self.increment(packet)

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError)

    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs,self.resultDifferences))/len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences))


    def visualizeData(self):

        if self.makePlots == False:
            return

        directory =''
        if self.savePlots:
            directory = os.path.dirname(self.saveLocation)
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.plot(self.arrivals[100:800])
        plt.plot(self.results[100:800], color='red')
        plt.ylabel('Rats' + self.name)

        if self.savePlots:
            plt.savefig(directory + self.prefix + self.name + "_Predictions" + ".png")

        plt.show()

        fig4, ax4 = plt.subplots()
        ax4.set_title('Hide Outlier Points')
        ax4.boxplot(self.resultDifferences, showfliers=False)

        if self.savePlots:
            fig4.savefig(directory + self.prefix + self.name + "_Boxplot_Error" + ".png")

        fig4.show()

        plt.plot(self.resultDifferences[100:300], color='red')
        plt.ylabel('Rats' + self.name)

        if self.savePlots:
            plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

        plt.show()


#######################################################
#######################################################


########
# optimized armax model to eliminate increasing errors due to outliers.
########


class ARStreamMax(BaseStream):

    def __init__(self,name,times, window):
        super(ARStreamMax, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1
        self.windowArrival = window + 2
        self.trendPredictions = []
        self.trendAccuracy = 0

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = AR(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]


        minVal = 0
        maxVal =0
        if len(self.predictedArrivals) < 4:
            minVal = min(self.predictedArrivals[-self.windowArrival:])
            maxVal = max(self.predictedArrivals[-self.windowArrival:])
        else:
            cut = self.predictedArrivals[-self.windowArrival:]
            cut.sort()
            minVal = cut[1]
            maxVal= cut[-2]

        if self.prediction < minVal:
            self.prediction = minVal
        elif self.prediction > maxVal:
            self.prediction = maxVal
        else:
            return




    def initialConfiguration(self,lat):
        print('Data for AR model with TW: ', self.window)
        self.last_arrival_time = lat
        self.index = 0
        self.prediction = lat

    def increment(self,rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.update()


    def runSimulation(self):

        for x in range(1):
            packet = self.arrivals[x]
            #self.increment(packet)
            self.initialConfiguration(packet)
            self.index = 10

        for packet in self.arrivals[10:]:
            #print('making prediction for: ', packet)
            #print("Packet ", packet)
            previousArrival = -100
            if len(self.predictedArrivals) > 1:
                previousArrival = self.predictedArrivals[-1]
            self.predictedArrivals.append(packet)
            self.resultDifferences.append(self.prediction - packet)
            self.results.append(self.prediction)



            if previousArrival != -100:

                actualTrend = packet - previousArrival
                predTrend =  packet - self.prediction

                #print("Prev: ", previousArrival, " Curr: ", packet, " Actual-Pred trand ", actualTrend, " , ", predTrend )

                if actualTrend * predTrend < 0:
                    self.trendPredictions.append(0)
                else:
                    self.trendPredictions.append(1)

                #print(self.trendPredictions[-1])


            self.increment(packet)

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError , "\t\t TrendAcc: ", self.trendAccuracy)

    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs,self.resultDifferences))/len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences))

        self.trendAccuracy = sum(self.trendPredictions)/len(self.trendPredictions)

########
# optimized armax model to eliminate increasing errors due to outliers.
########

class ARStreamAltMax(BaseStream):

    def __init__(self, name, times, window):
        super(ARStreamAltMax, self).__init__(name, times)
        self.window = 500
        self.model = -1
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

    # updates SAT
    def update(self):
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
        else:
            return

    def initialConfiguration(self, lat):
        print('Data for AR model with TW: ', self.window)
        self.last_arrival_time = lat
        self.index = 0
        self.prediction = lat

        file = './Alternating/alt_model.h5'

        from keras.models import load_model
        self.modelAlt = load_model(file)

    def increment(self, rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.update()

    def runSimulation(self):

        for x in range(1):
            packet = self.arrivals[x]
            # self.increment(packet)
            self.initialConfiguration(packet)
            self.index = 10

        for packet in self.arrivals[10:]:
            # print('making prediction for: ', packet)
            # print("Packet ", packet)
            previousArrival = -100
            if len(self.predictedArrivals) > 1:
                previousArrival = self.predictedArrivals[-1]
            self.predictedArrivals.append(packet)
            self.resultDifferences.append(self.prediction - packet)
            self.results.append(self.prediction)

            self.predictAltenating()

            if previousArrival != -100:

                ARdirection = 1 if self.prediction > previousArrival else 0

                actualTrend = 1 if packet - previousArrival > 0 else 0
                predTrend = 1 if self.prediction - previousArrival > 0 else 0

                # print("Prev: ", previousArrival, " Curr: ", packet, " Actual-Pred trand ", actualTrend, " , ", predTrend )

                if actualTrend != self.altPred: # false predictions
                    self.NNTrendPredictions.append(0)
                else:                           # correct prediction
                    self.NNTrendPredictions.append(1)

                if actualTrend != predTrend: # false predictions
                    self.trendPredictions.append(0)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(0)
                else:                           # correct prediction
                    self.trendPredictions.append(1)
                    if predTrend == int(self.altPred):
                        self.sameTrendPredictions.append(1)


                # print(self.trendPredictions[-1])

            self.increment(packet)

        self.calculateMetrics()

        print('MAE: ', self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ",
              self.meanRootSquaredError, "\nTrendAcc: ", self.trendAccuracy, '\t\t NNACC: ', self.NNTrendAcc, '\t\t Same pred ACC: ', self.sameTrendAccuracy ,', for ', len(self.sameTrendPredictions))

    def predictAltenating(self):
        # print(self.index)
        # print(self.arrivals[self.index])
        # print(self.arrivals[self.index-5:self.index])

        X = [self.arrivals[self.index-5:self.index]]
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        pred = self.modelAlt.predict(X)[0]
        self.altPred = round(pred[0])



    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs, self.resultDifferences)) / len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x: x * x, self.resultDifferences)) / len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(
            sum(map(lambda x: x * x, self.resultDifferences)) / len(self.resultDifferences))

        self.trendAccuracy = sum(self.trendPredictions) / len(self.trendPredictions)
        self.sameTrendAccuracy = sum(self.sameTrendPredictions) / len(self.sameTrendPredictions)
        self.NNTrendAcc = sum(self.NNTrendPredictions) / len(self.NNTrendPredictions)

