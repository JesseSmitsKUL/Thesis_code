from baseStream import BaseStream
from copy import deepcopy

# simulation model to estimate minimal K mapping

class SimARMAXKhronos(BaseStream):

	def __init__(self, name, times, window, save, predictions, variances, kval):
		super(SimARMAXKhronos, self).__init__(name, deepcopy(times))
		self.alpha = 7.0 / 8.0
		self.beta = 3.0 / 4
		self.variance = -1
		self.meanPE = []
		self.variancePE = []
		self.stdPE = []
		self.rmsPE = []
		self.medianPE = []

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

		self.k = kval
		self.predictions = deepcopy(predictions)
		self.variances = deepcopy(variances)
		self.comp_window_size = 600 # smaller window size to make it more robust since the variations in missed timeouts is smaller


	def initialConfigurationSim(self):

		self.arrivals = self.arrivals[-1800:]
		self.predictions = self.predictions[-1800:]
		self.variances = self.variances[-1800:]

		self.timeouts = []
		self.accuracies_constraints = []
		self.moving_average_accuracy_constraints = []
		self.moving_average_achieved_completeness = []
		self.moving_average_achieved_completenesses = []
		self.constraint_violations = []
		self.timeoutDifferences= []
		self.meanSquaredError = []
		self.meanRootSquaredError = []
		self.meanAbsError = []
		self.timeoutDifferences = []


	def runSimulation(self):

		self.initialConfigurationSim()

		for packet in self.arrivals:
			self.predictedArrivals.append(packet)
			self.resultDifferences.append(self.predictions[0] - packet)

			timeout = round(self.predictions[0] + (self.k * self.variances[0]), 3)
			self.timeouts.append(timeout)
			ontime = 1 if packet < timeout else 0

			if len(self.accuracies_constraints) < self.comp_window_size:
				self.accuracies_constraints.append(ontime)
			else:
				self.accuracies_constraints = self.accuracies_constraints[1:] + [ontime]


			self.updateMovingAverageAccCons()

			del self.predictions[0]
			del self.variances[0]


	def updateMovingAverageAccCons(self):
		length = len(self.accuracies_constraints)
		average = sum(self.accuracies_constraints)
		moving_av = round(average / length, 2)

		if len(self.moving_average_accuracy_constraints) < self.comp_window_size:
			self.moving_average_accuracy_constraints.append(moving_av)
		else:
			self.moving_average_accuracy_constraints = self.moving_average_accuracy_constraints[1:] + [moving_av]

	def updateCompleteness(self):
		key = 0
		count_above_completeness = sum(mov_av >= key for mov_av in self.moving_average_accuracy_constraints[key])
		percentage = (count_above_completeness * 100) / len(self.moving_average_accuracy_constraints[key])
		percentage = round(percentage, 2)

		if len(self.moving_average_achieved_completeness[key]) < self.comp_window_size:
			self.moving_average_achieved_completeness[key].append(percentage)
		else:
			self.moving_average_achieved_completeness[key] = self.moving_average_achieved_completeness[key][1:] + [percentage]
