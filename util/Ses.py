from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from baseStream import BaseStream

class SeSStream(BaseStream):

    def __init__(self,name,times):
        super(SeSStream, self).__init__(name, times)
        self.window = 100
        self.model = -1
        self.index = -1
        self.startIndex = 2

    # updates SAT
    def update(self):
        begin = max(0,self.index-self.window)
        data = self.arrivals[begin:self.index]
        # fit model
        model = SimpleExpSmoothing(data)
        model_fit = model.fit()
        self.model = model_fit
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

        if self.savePlots:
            directory = os.path.dirname(self.saveLocation)
            if not os.path.exists(directory):
                os.makedirs(directory)

        plt.plot(self.arrivals[1400:1800])
        plt.plot(self.results[1400:1800], color='red')
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
