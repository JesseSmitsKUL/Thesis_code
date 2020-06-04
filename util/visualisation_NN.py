import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from Khronos import Khronos
from expSmoothedAvgStream import ExpSmoothedAvgStream
from statisticalTests import test_stationarity
from newKhronos import MyKhronos
from neuralKhronos import NNKhronos, ARMAXKhronos
from statisticalStream import ARStream, MAStream, AMAStream, SeSStream, HoltStream, HoltWintersStream

import warnings


CUTOFF = 0.7

makePlots = True
savePlots = False
timeWindows = True
saveLocation = "./Visuals/ExpSmoothing/"
type = "inc"
filePath = "../Datasets/rats_" + type + ".csv"
prefix = "/ExpSm_alpha_125"



data = dict()
with open(filePath,'r', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=';')
    line_count = 0
    for row in csv_reader:
        line = str(row['Series'])
        linesplit = line.split(' ')[2:]
        key = linesplit[0][:-1] + "_" + linesplit[2]
        key = key[21:]
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
print('hello')

loops = 0
for key in datakeys:

    stream = data[key]

    cutoff = round(len(stream) * CUTOFF)

    print('\t' + key)

    keys = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]    #0.3,0.5,0.6,0.7,0.8,0.9]

    #streamModel = NNKhronos(key,stream,'cnn_gru_test',CUTOFF, type)
    #streamModel = ARMAXKhronos(key,stream,10,type)
    streamModel = Khronos(key,stream[:1000])
    streamModel.setKeys(keys)
    streamModel.setVisualConfiguration(makePlots)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        streamModel.runSimulation()
    streamModel.visualizeData()
    streams.append(streamModel)


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