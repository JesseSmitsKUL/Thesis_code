import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from Khronos import Khronos
from expSmoothedAvgStream import ExpSmoothedAvgStream
from statisticalTests import test_stationarity
from newKhronos import MyKhronos
from statisticalStream import ARStream, MAStream, AMAStream, SeSStream, HoltStream, HoltWintersStream
from ARStream import ARStreamMax, ARStreamAltMax

import warnings




makePlots = True
savePlots = False
timeWindows = True
saveLocation = "./Visuals/ExpSmoothing/"
filePath = "../Datasets/rats_static.csv"
prefix = "/ExpSm_alpha_125"

Methods = ['smoothed']



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

keys = []

loops = 0
for key in datakeys[1:2]:
    stream = data[key]


    #test_stationarity(stream)
    #continue

    cutoff = round(len(stream) * 0.8)

    print('\t' + key)

    # keys = [0.6,0.7,0.8,0.9,0.95,0.99]
    keys = [0.6]

    streamModel = ARStreamAltMax(key,stream, 10)
    #streamModel.setKeys(keys)
    streamModel.setVisualConfiguration(makePlots)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        streamModel.runSimulation()
    streamModel.visualizeData()
    streams.append(streamModel)


if timeWindows:

    print('PREDICTION TIME WINDOW ESTIMATIONS')


    totalAE = [0 for x in keys]
    totalSE = [0 for x in keys]
    totalMRSE = [0 for x in keys]
    for s in streams:
        totalAE = [sum(x) for x in zip(totalAE,s.meanAbsError)]
        totalSE = [sum(x) for x in zip(totalAE,s.meanSquaredError)]
        totalMRSE = [sum(x) for x in zip(totalAE,s.meanRootSquaredError)]

        # if s.name[-4:] == "dabe":
        #     print('items  ',([x for x in s.results if x > 80]))
        #     print('sum of squares ', sum([x**2 for x in s.results if x > 80]))
        #     print('info ', len(s.results))

        for x in range(len(keys)):
            print('Evaluation key: ', keys[x])
            print("\n\nTotal Mean Abs Error " + str(totalAE[x] / len(streams)))
            print("Total Mean Squared Error " + str(totalSE[x] / len(streams)))
            print("Total Mean Root Squared Error " + str(totalMRSE[x] / len(streams)))

else:

    print('ARRIVAL TIME ESTIMATIONS')

    totalAE = 0
    totalSE = 0
    totalMRSE = 0
    for s in streams:
        totalAE += s.meanAbsError
        totalSE += s.meanSquaredError
        totalMRSE += s.meanRootSquaredError

        # if s.name[-4:] == "dabe":
        #     print('items  ',([x for x in s.results if x > 80]))
        #     print('sum of squares ', sum([x**2 for x in s.results if x > 80]))
        #     print('info ', len(s.results))

    print("\n\nTotal Mean Abs Error " + str(totalAE/len(streams)))
    print("Total Mean Squared Error " + str(totalSE/len(streams)))
    print("Total Mean Root Squared Error " + str(totalMRSE/len(streams)))
