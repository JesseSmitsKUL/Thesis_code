from baseStream import BaseStream
import matplotlib.pyplot as plt
import os
from math import sqrt


class ExpSmoothedAvgStream(BaseStream):

    def __init__(self,name,times):
        super(ExpSmoothedAvgStream, self).__init__(name, times)
        self.alpha = 7.0/8.0
        self.beta = 3.0/4

    # updates SAT
    def update(self):
        calc_smoothed_arrival_time = round(self.alpha * self.prediction + (1 - self.alpha) * self.last_arrival_time, 2)
        self.prediction = calc_smoothed_arrival_time

    def initialConfiguration(self,lat):
        self.prediction = lat
        self.last_arrival_time = lat

    def increment(self,rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.last_arrival_time = rat
            self.update()

    def runSimulation(self):

        for x in range(1):
            packet = self.arrivals[x]
            self.increment(packet)

        for packet in self.arrivals[1:]:
            #print("Packet ", packet)
            self.resultDifferences.append(self.prediction - packet)
            self.predictedArrivals.append(packet)
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

        plt.plot(self.arrivals, label='arrival times')
        plt.ylabel('arrival time (seconds)')
        plt.title('Arrival times in CPS')
        plt.show()

        for x in range(5):
            plt.plot(self.predictedArrivals[100+(x-1)*20:100+x*20], marker='o', label='arrival')
            plt.plot(self.results[100+(x-1)*20:100+x*20], marker='D', label='prediction')
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


class Khronos(BaseStream):

    def __init__(self,name,times):
        super(Khronos, self).__init__(name, times)
        self.alpha = 7.0/8.0
        self.beta = 3.0/4
        self.variance = -1
        self.constraints_to_K = {0.1: 0, 0.2: 0.1, 0.3: 0.6, 0.4: 1, 0.5: 1.2, 0.6: 1.4, 0.7: 2, 0.8: 2.8, 0.9: 4.9, 1:300}

    def setKeys(self, keys):
        self.constrains = keys

    # updates SAT
    def update(self):
        self.variance = round(self.beta * self.variance + (1 - self.beta) * abs(self.prediction - self.last_arrival_time), 2)
        self.prediction = round(self.alpha * self.prediction + (1 - self.alpha) * self.last_arrival_time, 2)

        print('JACOB: ',self.prediction, ' ', self.variance)

        for key in self.constrains:
            new_timeout = round(self.prediction + self.constraints_to_K[key] * self.variance,3)
            self.timeouts[key].append(new_timeout)
        print('timeouts:' , self.timeouts)



    def initialConfiguration(self,lat):
        self.prediction = lat
        self.variance = float(lat)/2
        self.last_arrival_time = lat
        self.timeouts = dict()
        for key in self.constrains:
            self.timeouts[key] = [2*lat]
            self.accuracies_constraints[key] = []
            self.moving_average_accuracy_constraints[key] = []
            self.moving_average_achieved_completeness[key] = []
            self.moving_average_achieved_completenesses[key] = []
            self.constraint_violations[key] = []
            self.timeoutDifferences[key] = []
            self.meanSquaredError = []
            self.meanRootSquaredError = []
            self.meanAbsError = []

    def increment(self,rat):
        if self.last_arrival_time == -1:
            self.initialConfiguration(rat)
        else:
            self.last_arrival_time = rat
            self.update()

    def runSimulation(self):

        packet = self.arrivals[0]
        self.increment(packet)
        print('initial: ', packet)


        for packet in self.arrivals[1:]:
            for key in self.constrains:
                timeout = self.timeouts[key][-1]
                ontime = 1 if packet < timeout else 0

                print('Packet arrived at: ',packet, ", with timeout: ", timeout, ", ontime: ", ontime)

                if len(self.accuracies_constraints[key]) < self.comp_window_size:
                    self.accuracies_constraints[key].append(ontime)
                else:
                    self.accuracies_constraints[key] = self.accuracies_constraints[key][1:] + [ontime]
                print('acc_cons: ', self.accuracies_constraints[key])


                self.timeoutDifferences[key].append(round(timeout-packet,2))
                print('differences:', self.timeoutDifferences[key])

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
            print('updateMovinceAvAccCons:', self.moving_average_accuracy_constraints[key])

    def updateCompleteness(self):
        for key in self.constrains:
            count_above_completeness = sum(mov_av >= key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness*100)/len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage,2)

            if len(self.moving_average_achieved_completeness[key]) < self.comp_window_size:
                self.moving_average_achieved_completeness[key].append(percentage)
            else:
                self.moving_average_achieved_completeness[key] = self.moving_average_achieved_completeness[key][1:] + [percentage]
            print('UpdateCompleteness: ',  self.moving_average_achieved_completeness[key])

    def updateViolation(self):
        # for key in self.constrains:
            # count_below_completeness = sum(completeness < key*100 for completeness in self.moving_average_achieved_completeness[key])
            # percentage = (count_below_completeness*100)/len(self.moving_average_achieved_completeness[key])
            # self.constraint_violations[key].append(percentage)
            # print('UpdateViolation: ', self.constraint_violations[key])
        for key in self.constrains:
            count_above_completeness = sum(
                mov_av < key for mov_av in self.moving_average_accuracy_constraints[key])
            percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
            percentage = round(percentage, 2)
            self.constraint_violations[key].append(percentage)
            print('UpdateViolation: ', self.constraint_violations[key])





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


        datapoints = 250
        for key in self.constrains:
            for x in range(2):
                plt.plot(self.predictedArrivals[400+(x-1)*datapoints:400+x*datapoints], label='arrival')
                plt.plot(self.timeouts[key][400+(x-1)*datapoints:400+x*datapoints], label='prediction')
                plt.legend()
                plt.ylabel('window' + self.name)

                if self.savePlots:
                    plt.savefig(directory + self.prefix + self.name + "_Predictions" + ".png")

                plt.show()

        fig, ax = plt.subplots()
        ax.boxplot(self.timeoutDifferences.values())
        ax.set_xticklabels(self.timeoutDifferences.keys())
        fig.show()

        for key in self.constrains:
            fig4, ax4 = plt.subplots()
            ax4.set_title('Hide Outlier Points')
            ax4.boxplot(self.timeoutDifferences[key], showfliers=False)

            if self.savePlots:
                fig4.savefig(directory + self.prefix + self.name + "_Boxplot_Error" + ".png")

            fig4.show()

        for key in self.constrains:
            plt.plot(self.timeoutDifferences[key][100:300], color='red')
            plt.axhline(y=0, color='green', linestyle='-')
            plt.ylabel('differences' + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()

        for key in self.constrains:
            plt.plot(self.constraint_violations[key][0:100], color='red')
            plt.ylabel('Constraint violation (\%) ' + self.name)

            if self.savePlots:
                plt.savefig(directory + self.prefix + self.name + "Error" + ".png")

            plt.show()
