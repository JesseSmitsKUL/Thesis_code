from baseStream import BaseStream
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
import datetime

from random import random

import matplotlib.pyplot as plt
import os
from math import sqrt


class ARStream(BaseStream):

    def __init__(self,name,times):
        super(ARStream, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1

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


class MAStream(BaseStream):

    def __init__(self,name,times):
        super(MAStream, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = ARMA(data, order=(0, 1))
        model_fit = 0

        try:
            model_fit = model.fit(disp=False)
            passed = True
        except:
            return 0

        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]
        #print('upcominng prediction based of last: ', data[-1] , "   ", data[-5:])



    def initialConfiguration(self,lat):
        print('Data for Moving Avg model with TW: ', self.window)
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
            self.initialConfiguration(packet)
            self.index = 10

        for packet in self.arrivals[10:]:
            if self.index % 150 == 0: print(self.index)
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




class AMAStream(BaseStream):

    def __init__(self,name,times):
        super(AMAStream, self).__init__(name, times)
        self.window = 10
        self.model = -1
        self.index = -1

    # updates SAT
    def update(self):
        print(self.index)
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = ARMA(data, order=(2, 1))
        model_fit = 0
        model_fit = model.fit(disp=False,start_ar_lags=6)
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]
        #print('upcominng prediction based of last: ', data[-1] , "   ", data[-5:])



    def initialConfiguration(self,lat):
        print('Data for AUTO Moving Avg model with TW: ', self.window)
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
            self.initialConfiguration(packet)
            self.index = 10

        for packet in self.arrivals[10:]:
            if self.index % 150 == 0: print(self.index)
            #print('making prediction for: ', packet)
            #print("Packet ", packet)
            self.resultDifferences.append(self.prediction - packet)
            self.results.append(self.prediction)
            self.increment(packet)

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError)

    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs,self.resultDifferences))/len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences))




class SeSStream(BaseStream):

    def __init__(self,name,times):
        super(SeSStream, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1
        self.startIndex = 10

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = SimpleExpSmoothing(data)
        model_fit = model.fit()
        self.model = model_fit
        self.smoothedValues.append(self.model.params['smoothing_level'])
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]
        #print('upcominng prediction based of last: ', data[-1] , "   ", data[-5:])



    def initialConfiguration(self,lat):
        print('Data for AR model with TW: ', self.window)
        self.last_arrival_time = lat
        self.index = self.startIndex
        self.prediction = lat

    def increment(self,rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.index += 1
            self.last_arrival_time = rat
            self.update()


    def runSimulation(self):

        self.initialConfiguration(self.arrivals[0])

        now = datetime.datetime.now()
        for packet in self.arrivals[self.startIndex:]:
            #print('making prediction for: ', packet)
            #print("Packet ", packet)
            self.resultDifferences.append(self.prediction - packet)
            self.predictedArrivals.append(packet)
            self.results.append(self.prediction)
            self.increment(packet)
            afterCalc = datetime.datetime.now()
            dif = afterCalc-now
            #print('Time = ',dif.seconds,':',dif.microseconds)
            now = afterCalc

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError)

    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs,self.resultDifferences))/len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences))

    def visualizeData(self):

        if self.makePlots == False:
            return

        if self.savePlots:
            directory = os.path.dirname(self.saveLocation)
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.plot(self.arrivals)
        plt.show()

        print(self.smoothedValues[:200])
        plt.plot(self.smoothedValues)
        plt.show()

        return

class HoltStream(BaseStream):

    def __init__(self,name,times):
        super(HoltStream, self).__init__(name, times)
        self.window = 500
        self.model = -1
        self.index = -1

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = Holt(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]
        #print('upcominng prediction based of last: ', data[-1] , "   ", data[-5:])



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



class HoltWintersStream(BaseStream):

    def __init__(self,name,times):
        super(HoltWintersStream, self).__init__(name, times)
        self.window = 100
        self.model = -1
        self.index = -1

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = ExponentialSmoothing(data)
        model_fit = model.fit()
        self.model = model_fit
        # make prediction
        self.prediction = model_fit.predict(len(data), len(data))[0]
        #print('upcominng prediction based of last: ', data[-1] , "   ", data[-5:])



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
            self.results.append(0)
            self.initialConfiguration(packet)
            self.index = 20

        for packet in self.arrivals[20:]:
            #print('making prediction for: ', packet)
            #print("Packet ", packet)
            self.resultDifferences.append(self.prediction - packet)
            self.results.append(self.prediction)
            self.increment(packet)

        self.calculateMetrics()

        print('MAE: ' , self.meanAbsError, "\t\t MSE: ", self.meanSquaredError, "\t\t MRSE: ", self.meanRootSquaredError)

    def calculateMetrics(self):
        self.meanAbsError = sum(map(abs,self.resultDifferences))/len(self.resultDifferences)
        self.meanSquaredError = sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences)
        self.meanRootSquaredError = sqrt(sum(map(lambda x:x*x,self.resultDifferences))/len(self.resultDifferences))