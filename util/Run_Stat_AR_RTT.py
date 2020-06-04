import csv
import os
import pandas as pd
from statisticalTests import test_stationarity
from ARStream import ARStreamMax, ARStreamSES
from statisticalStream import ARStream
from matplotlib import pyplot

from datetime import datetime
from createNN import *
import numpy as np

import warnings

TRAIN_TEST_SPLIT = 0.7



def run(model, location = ''):


    # evaluate the model
    y_pred = model.results

    dif = model.resultDifferences
    np.savetxt(location + "Errors.txt", dif)
    np.save(location + "ErrorNP", dif)

    pyplot.subplot(111)
    pyplot.title('prediction')
    pyplot.plot(model.predictedArrivals, marker='o', label='arrival')
    pyplot.plot(y_pred, label='prediction')
    pyplot.legend()
    pyplot.savefig(location + "AllPredictions.jpg")
    #pyplot.show()
    pyplot.close()


    for x in range(10): #edit
        # plot loss during training
        pyplot.subplot(111)
        pyplot.title('prediction')
        pyplot.plot(model.predictedArrivals[x*20:(x+1)*20], marker='o', label='arrival')
        pyplot.plot(y_pred[x*20:(x+1)*20], marker='D', label='prediction')
        pyplot.legend()

        if location != '':
            # datetime object containing current date and time
            now = datetime.now()

            # dd/mm/YY H:M:S
            postfix = now.strftime("%H-%M-%S_" + str(x) +"_predictions.jpg")
            pyplot.savefig(location+postfix)
        #pyplot.show()
        pyplot.close()

def evaluateModel(model, performanceStream = None, modelName = ""):


    if performanceStream:
        performanceStream.write('---------------------------------------------------------------\n' + modelName + '\n+---------------------------------------------------------------\n')
        performanceStream.write('MAE: ' + str(model.meanAbsError) + "\t\t MSE: " + str(model.meanSquaredError) + "\t\t MRSE: " + str(model.meanRootSquaredError) + "\n")



def singleRun():
    filePath = "../Datasets/rats_inc.csv"

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

    for key in datakeys:
        set = data[key]

        model = ARStream("khronos", set)
        model.runSimulation()

        break

def evaluateRuns():
    rootFolder = './Stat_AR_Evaluation/'

    print(rootFolder)

    csv_files = {'inc': "../Datasets/rats_inc.csv", 'static': "../Datasets/rats_static.csv"}

    models = ["ARMAX10"]
    # models = ['AR',"ARMAX10" ,"ARMAX20" ,"ARMAX40" , "ARSES10" , "ARSES20" , "ARSES40"]

    # dir time
    #   dir static
    #   dir incremental
    #       dir stream
    #           performance.txt
    #            dir model
    #               plots + data

    directory = os.path.dirname(rootFolder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S/")

    timeRoot = rootFolder + dt_string
    print(timeRoot)
    directory = os.path.dirname(timeRoot)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file in csv_files.keys():

        streamRoot = timeRoot + file + '/'
        print(streamRoot)
        directory = os.path.dirname(streamRoot)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = dict()
        with open(csv_files[file], 'r', encoding='utf-8-sig') as csvfile:
            csv_reader = csv.DictReader(csvfile, delimiter=';')
            line_count = 0
            for row in csv_reader:
                line = str(row['Series'])
                linesplit = line.split(' ')[2:]
                key = linesplit[0][:-1] + "_" + linesplit[2]
                key = key[21:].replace("/","_").replace(',',"")

                if key in data:
                    times = data[key]
                    value = float(row['Value'])
                    if len(times) > 0 and value == times[-1]:
                        value += 0.001
                    data[key].append(value)
                else:
                    data[key] = []

        datakeys = [*data]
        datakeys.sort()
        streams = []

        for key in datakeys:
            set = data[key]

            streamname = key
            streamNameRoot = streamRoot + streamname + '/'
            directory = os.path.dirname(streamNameRoot)
            if not os.path.exists(directory):
                os.makedirs(directory)

            file_performance = open(streamNameRoot + 'performance.txt', 'w')

            #cutoff = round(len(set) * TRAIN_TEST_SPLIT)

            for modelname in models:

                model = -1
                history = -1

                if modelname == 'AR':
                    model = ARStream("Ar", set)
                elif modelname == 'ARMAX10':
                    model = ARStreamMax("Armax", set, 10)
                elif modelname == 'ARMAX20':
                    model = ARStreamMax("Armax", set, 20)
                elif modelname == 'ARMAX40':
                    model = ARStreamMax("Armax", set, 40)
                elif modelname == 'ARSES10':
                    model = ARStreamSES("Arses", set, 10)
                elif modelname == 'ARSES20':
                    model = ARStreamSES("Arses", set, 20)
                elif modelname == 'ARSES40':
                    model = ARStreamSES("Arses", set, 40)
                else:
                    print("Cannot run the given model: ", model)
                    continue

                model.runSimulation()

                modelRoot = streamNameRoot + modelname + '/'
                directory = os.path.dirname(modelRoot)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                evaluateModel(model,
                              performanceStream=file_performance, modelName=modelname)
                run(model, modelRoot)

evaluateRuns()
# singleRun()



