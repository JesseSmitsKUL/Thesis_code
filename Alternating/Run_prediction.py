import csv
import numpy as np
from numpy import array

from Alternating.createBinaryNN import transformAltStream, createAltModelLSTM, createSVM

import warnings

TRAIN_TEST_SPLIT = 0.7
LEN_STREAM = 10

csv_files = {'constraints': "../../Datasets/Heterogeneity/Constraints/rats_static_60.csv"}

x_train, y_train, y_test, x_test = [], [], [], []

for file in csv_files.keys():

    # streamRoot = timeRoot + file + '/'
    # print(streamRoot)
    # directory = os.path.dirname(streamRoot)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    data = dict()
    with open(csv_files[file], 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';')
        line_count = 0
        for row in csv_reader:
            line = str(row['Series'])
            linesplit = line.split(' ')[2:]
            key = linesplit[0][:-1] + "_" + linesplit[2]
            key = key[21:].replace("/", "_").replace(',', "")

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

        cutoff = round(len(set) * TRAIN_TEST_SPLIT)

        x_set, y_set = transformAltStream(set[:cutoff], LEN_STREAM)
        x_settest, y_settest = transformAltStream(set[cutoff:], LEN_STREAM)

        x_train = x_set
        y_train = y_set
        x_test =  x_settest
        y_test =  y_settest


        print(len(x_train))
        print(len(x_test))

        streamModel, hist = createSVM(x_train, y_train)

        #streamModel.summary()

        x_test2 = array(x_test)
        y_test2 = array(y_test)
        x_train2 = array(x_train)
        y_train2 = array(y_train)

        #x_test2 = x_test2.reshape((x_test2.shape[0], x_test2.shape[1], 1))
        #x_train2 = x_train2.reshape((x_train2.shape[0], x_train2.shape[1], 1))

        from sklearn import metrics
        # Model Accuracy: how often is the classifier correct?

        #Predict the response for test dataset
        y_pred = streamModel.predict(x_test2)
        print("Accuracy:",metrics.accuracy_score(y_pred, y_test2))

# evaluate the model
# scoresTrain = streamModel.evaluate(x_train2, y_train2)
# scoresTest = streamModel.evaluate(x_test2, y_test2)
# print("TRAIN %s: %.2f%%" % (streamModel.metrics_names[1], scoresTrain[1] * 100))
# print("TEST %s: %.2f%%" % (streamModel.metrics_names[1], scoresTest[1] * 100))
