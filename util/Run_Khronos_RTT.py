import csv
from matplotlib import pyplot
import os

from expSmoothedAvgStream import ExpSmoothedAvgStream

from datetime import datetime
from createNN import *
import numpy as np

import warnings

# define stream
LEN_STREAM = 30
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

def evaluateModel(model, performanceStream = None):


    if performanceStream:
        performanceStream.write('---------------------------------------------------------------\n' + "Khronos" + '\n+---------------------------------------------------------------\n')
        performanceStream.write('MAE: ' + str(model.meanAbsError) + "\t\t MSE: " + str(model.meanSquaredError) + "\t\t MRSE: " + str(model.meanRootSquaredError))



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

        model = ExpSmoothedAvgStream("khronos", set)
        model.runSimulation()

        evaluateModel(model)
        run(model)

        break

def evaluateRuns():
    rootFolder = './Khronos_Evaluation/'

    csv_files = {'inc': "../Datasets/rats_inc.csv", 'static': "../Datasets/rats_static.csv"}

    directory = os.path.dirname(rootFolder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S/")

    timeRoot = rootFolder + dt_string
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

            modelRoot = streamNameRoot

            cutoff = round(len(set) * TRAIN_TEST_SPLIT)


            model = ExpSmoothedAvgStream("khronos", set[cutoff:])
            model.runSimulation()

            evaluateModel(model)
            run(model)


            evaluateModel(model, performanceStream=file_performance)
            run(model,modelRoot)


evaluateRuns()
# singleRun()



