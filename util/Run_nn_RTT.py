import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from expSmoothedAvgStream import ExpSmoothedAvgStream
from statisticalTests import test_stationarity
from statisticalStream import ARStream, MAStream, AMAStream, SeSStream, HoltStream, HoltWintersStream
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM, Dropout,TimeDistributed, GRU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from datetime import datetime
from createNN import *
import numpy as np

import warnings

# define stream
LEN_STREAM = 30
TRAIN_TEST_SPLIT = 0.7



def run(model,x_test,y_test, location = ''):
    from matplotlib import pyplot
    x_test2 = array(x_test)
    y_test2 = array(y_test)
    x_test2 = x_test2.reshape((x_test2.shape[0], x_test2.shape[1], 1))

    # evaluate the model
    y_pred = model.predict(x_test2)

    print("DIFFERENCE")
    print(y_pred.flatten())
    print(y_test2)
    dif = np.subtract(y_test2,y_pred.flatten())
    print(dif)

    np.savetxt(location + "Errors.txt", dif)
    np.save(location + "ErrorNP", dif)

    pyplot.subplot(111)
    pyplot.title('prediction')
    pyplot.plot(y_test, marker='o', label='arrival')
    pyplot.plot(y_pred, label='prediction')
    pyplot.legend()
    pyplot.savefig(location + "AllPredictions.jpg")
    #pyplot.show()
    pyplot.close()


    for x in range(10): #edit
        # plot loss during training
        pyplot.subplot(111)
        pyplot.title('prediction')
        pyplot.plot(y_test[x*20:(x+1)*20], marker='o', label='arrival')
        pyplot.plot(y_pred[x*20:(x+1)*20], marker='D', label='prediction')
        pyplot.legend()
        # plot mse during training
        # pyplot.subplot(212)
        # pyplot.title('Mean Abs Error')
        # pyplot.plot(history.history['mean_absolute_error'], label='train')
        # pyplot.plot(history.history['val_mean_absolute_error'], label='test')
        # pyplot.legend()

        if location != '':
            # datetime object containing current date and time
            now = datetime.now()

            # dd/mm/YY H:M:S
            postfix = now.strftime("%H-%M-%S_" + str(x) +"_predictions.jpg")
            pyplot.savefig(location+postfix)
        #pyplot.show()
        pyplot.close()

def evaluateModel(model,history,x_test,y_test, x_train, y_train, location = '', performanceStream = None , modelName = ''):
    from matplotlib import pyplot
    x_test2 = array(x_test)
    y_test2 = array(y_test)
    x_train2 = array(x_train)
    y_train2 = array(y_train)

    model.summary()

    x_test2  = x_test2.reshape((x_test2.shape[0], x_test2.shape[1], 1))
    x_train2 = x_train2.reshape((x_train2.shape[0], x_train2.shape[1], 1))

    # evaluate the model
    train_mae, train_mse = model.evaluate(x_train2, y_train2)
    test_mae, test_mse = model.evaluate(x_test2, y_test2)

    print('Train MAE: %.3f, Train MSE: %.3f' % (train_mae, train_mse))
    print('Test MAE: %.3f, Test MSE: %.3f' % (test_mae, test_mse))

    if performanceStream:
        performanceStream.write('---------------------------------------------------------------\n' +modelName + '\n+---------------------------------------------------------------\n')
        performanceStream.write('Train MAE: %.3f, Train MSE: %.3f\n' % (train_mae, train_mse))
        performanceStream.write('Test MAE: %.3f, Test MSE: %.3f\n\n' % (test_mae, test_mse))

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], marker='o', label='train')
    pyplot.plot(history.history['val_loss'], marker='D', label='test')
    pyplot.legend()
    # plot mse during training
    # pyplot.subplot(212)
    # pyplot.title('Mean Abs Error')
    # pyplot.plot(history.history['mean_absolute_error'], label='train')
    # pyplot.plot(history.history['val_mean_absolute_error'], label='test')
    # pyplot.legend()

    if location != ':':
        pyplot.savefig(location + 'epoch_loss.jpg')
    #pyplot.show()
    pyplot.close()


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
        cutoff = round(len(set)*TRAIN_TEST_SPLIT)

        # plt.plot(set)
        # plt.show()
        # continue
        #
        # preset = [0 for x in range(cutoff)]
        # plt.plot(set[:cutoff])
        # plt.plot(preset + set[cutoff:])
        # plt.show()
        # continue

        x_train,y_train = transformStream(set[:cutoff],LEN_STREAM)
        x_test, y_test  = transformStream(set[cutoff:],LEN_STREAM)
        model, history = createModelCNN_GRU_TEST(x_train,y_train)

        model.save('savedModelgru.h5')

        evaluateModel(model,history,x_test,y_test,x_train,y_train)
        print('\n###############  ' + key + '  ################\n')
        run(model,x_test,y_test)

        break

def evaluateRuns():
    rootFolder = './NN_Evalutaion/'

    print(rootFolder)

    csv_files = {'inc': "../Datasets/rats_inc.csv", 'static': "../Datasets/rats_static.csv"}

    models = ['cnn_gru_test', 'cnn', 'cnn2', 'gru','gru2', 'cnn_lstm', 'cnn_gru', 'cnn_lstm2', 'cnn_gru2']

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

            cutoff = round(len(set) * TRAIN_TEST_SPLIT)

            x_train, y_train = transformStream(set[:cutoff], LEN_STREAM)
            x_test, y_test = transformStream(set[cutoff:], LEN_STREAM)

            for modelname in models:

                model = -1
                history = -1

                if modelname == 'cnn':
                    model, history = createModelCNN(x_train, y_train)
                elif modelname == 'cnn2':
                    model, history = createModelCNN2(x_train, y_train)
                elif modelname == 'gru':
                    model, history = createModelGRU(x_train, y_train)
                elif modelname == 'gru2':
                    model, history = createModelGRU2(x_train, y_train)
                elif modelname == 'cnn_lstm':
                    model, history = createModelCNN_LSTM(x_train, y_train)
                elif modelname == 'cnn_lstm2':
                    model, history = createModelCNN_LSTM2(x_train, y_train)
                elif modelname == 'cnn_gru':
                    model, history = createModelCNN_GRU(x_train, y_train)
                elif modelname == 'cnn_gru2':
                    model, history = createModelCNN_GRU2(x_train, y_train)
                elif modelname == 'cnn_gru_test':
                    model, history = createModelCNN_GRU_TEST(x_train, y_train)
                else:
                    print("Cannot run the given model: ", model)
                    continue

                modelRoot = streamNameRoot + modelname + '/'
                directory = os.path.dirname(modelRoot)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                evaluateModel(model, history, x_test, y_test, x_train, y_train, location=modelRoot,
                              performanceStream=file_performance, modelName=modelname)
                run(model, x_test, y_test, modelRoot)

evaluateRuns()
# singleRun()



