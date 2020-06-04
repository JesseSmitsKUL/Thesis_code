import matplotlib.pyplot as plt


class BaseStream:

    def __init__(self,name,times):
        self.name = name
        self.last_arrival_time = -1
        self.prediction = -1
        self.arrivals = times
        self.results = []
        self.predictedArrivals = []
        self.resultDifferences = []
        self.stdAbsError = -1
        self.meanAbsError = -1
        self.meanSquaredError = -1
        self.meanRootSquaredError = -1
        self.smoothedValues = []
        self.comp_window_size = 1000  # for moving averages.
        self.constrains = []
        self.constraints_to_K = None
        self.timeouts = dict()
        self.timeoutDifferences = dict()

        self.accuracies_constraints = dict()
        self.moving_average_accuracies_constraints = dict()
        self.moving_average_accuracy_constraints = dict()
        self.moving_average_achieved_completeness = dict()
        self.moving_average_achieved_completenesses = dict()

        self.constraint_violations = dict()

        # visualisation information
        self.makePlots = True
        self.savePlots = False
        self.saveLocation = ""
        self.filePath = ""
        self.prefix = ""

        self.filewriter = None


    def setVisualConfiguration(self, makeplots = False, saveplots = False, saveloc = '', filepath = '', prefix = ''):
        self.makePlots = makeplots
        self.savePlots = saveplots
        self.saveLocation = saveloc
        self.filePath = filepath
        self.prefix = prefix

    def update(self):
        pass

    def initialConfiguration(self,lat):
        pass

    def increment(self,rat):
       pass

    def runSimulation(self):
        pass

    def calculateMetrics(selfs):
        pass


    def visualizeData(self):
        pass
