import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
from Evaluate_KValues.ARMAXKhronos_K import ARMAXKhronos_K
from Evaluate_KValues.evaluation import plotConstraints, plotTimeouts, plotTwoModel, plotdata
from Evaluate_KValues.parseScenarioData import parseNS, parseCNT, parseSensorData
from Evaluate_KValues.Khronos import Khronos

import warnings

# file used for a single run, mainly used for small experiments


CUTOFF = 0.7

evaluation = True
makePlots = True
savePlots = False
timeWindows = True
saveLocation = "./Visuals/ExpSmoothing/"
type = "static"
filePath = "../../Datasets/rats_" + type + ".csv"
filePath = "../../Datasets/Heterogeneity/Deployment/rats_thick.csv"



data = dict()
with open(filePath,'r', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line = str(row['Series'])
        linesplit = line.split(' ')[2:]
        print(linesplit)
        additionalInfo = "" if len(linesplit) < 3 else "_" + linesplit[2]
        key = linesplit[0][:-1] + additionalInfo
        key = key if len(linesplit) < 3 else key[21:]
        if key in data:
            times = data[key]
            value = float(row['Value'])
            if len(times) > 0 and value == times[-1]:
                value += 0.001
            data[key].append(value)
        else:
            data[key] = []
        #print(linesplit)
    #print(data)

datakeys = [*data]
datakeys.sort()
streams = []
streamsKhronos = []
print("val")
print(len(datakeys))

loops = 0

for key in datakeys[2:3]:
    stream = data[key]
    print(len(stream))
    #plotdata(stream)

    cutoff = round(len(stream) * CUTOFF)

    print('\t' + key)
    keys = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]  # ,0.9,0.95,0.99  #0.3,0.5,0.6,0.7,0.8,0.9]

    #streamModel = NNKhronos(key,stream,'cnn_gru_test',CUTOFF, type)
    #streamModel = ARMAXKhronos(key,stream,10,type)
    streamModel = ARMAXKhronos_K(key,stream, 10, type)
    streamModel.setKeys(keys)
    streamModel.setVisualConfiguration(makePlots)
    streamModel.makePlots = False
    streamModel.savePlots = False
    #streamModel.setScenario(parseCNT('sf')[key],parseSensorData('sf')[key])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        streamModel.runSimulation()
    #streamModel.visualizeData()
    streams.append(streamModel)

    if evaluation:
        #Khronosmodel = ARMAXKhronos_KRef(key,stream,10,type)
        Khronosmodel = Khronos(key,stream)
        Khronosmodel.setKeys(keys)
        Khronosmodel.setVisualConfiguration(makePlots)
        Khronosmodel.makePlots = False
        Khronosmodel.savePlots = False
        Khronosmodel.runSimulation()
        streamsKhronos.append(Khronosmodel)

    break


if timeWindows:

    print('PREDICTION TIME WINDOW ESTIMATIONS ', len(streams))


    totalAE = [0 for x in keys]
    totalSE = [0 for x in keys]
    totalMRSE = [0 for x in keys]
    for s in streams:
        totalAE = [sum(x) for x in zip(totalAE,s.meanAbsError)]
        totalSE = [sum(x) for x in zip(totalSE,s.meanSquaredError)]
        totalMRSE = [sum(x) for x in zip(totalMRSE,s.meanRootSquaredError)]

        # if s.name[-4:] == "dabe":
        #     print('items  ',([x for x in s.results if x > 80]))
        #     print('sum of squares ', sum([x**2 for x in s.results if x > 80]))
        #     print('info ', len(s.results))

    for x in range(len(keys)):
        print('Evaluation key: ', keys[x])
        print("\n\nTotal Mean Abs Error " + str(totalAE[x] / len(streams)))
        print("Total Mean Squared Error " + str(totalSE[x] / len(streams)))
        print("Total Mean Root Squared Error " + str(totalMRSE[x] / len(streams)))

if evaluation:
    plotConstraints(streams[0],streamsKhronos[0])
    plotTwoModel(streams[0],streamsKhronos[0],0.99)
    #plotTimeouts(streams[0], streamsKhronos[0])
