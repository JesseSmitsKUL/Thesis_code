import csv
import sys
sys.path.append("..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Evaluate_KValues.ARMAXKhronos_K import ARMAXKhronos_K
from matplotlib import pyplot
from Evaluate_KValues.Khronos import Khronos
from Evaluate_KValues.evaluation import plotTimeoutsSave, plotConstraints, plotTwoModelRun, plotTwomodelScenarioRun
from Evaluate_KValues.parseScenarioData import parseNS, parseCNT, parseSensorData
from Evaluate_KValues.ARMAXKhronos_K_es import ARMAXKhronos_K_es

from csv import writer
from datetime import datetime
import numpy as np

import warnings


# RUN  files are made to process all csv files and save the results given the models


keys = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def addData(file_name,armax,khronos,case,stream):
    index = 0
    for key in armax.constrains:
        append_list_as_row(file_name,[case,stream,key,armax.meanAbsError[index],armax.meanRootSquaredError[index],armax.MER[index],armax.trendAccuracy, armax.NNTrendAcc, armax.sameTrendAccuracy])
        append_list_as_row(file_name,[case, stream, key, khronos.meanAbsError[index], khronos.meanRootSquaredError[index], khronos.MER[index], '', '', ''])
        index += 1


def writetotalinfo(models, streams, file_performanceTotal):
	for model in models:
		file_performanceTotal.write("\n\n----------- " + model + " -----------\n")
		totalAE = [0 for x in keys]
		totalSE = [0 for x in keys]
		totalMRSE = [0 for x in keys]
		modelstreams = streams[model]
		for s in modelstreams:
			totalAE = [sum(x) for x in zip(totalAE, s.meanAbsError)]
			totalSE = [sum(x) for x in zip(totalSE, s.meanSquaredError)]
			totalMRSE = [sum(x) for x in zip(totalMRSE, s.meanRootSquaredError)]

		for x in range(len(keys)):
			file_performanceTotal.write('\n\nEvaluation key: ' + str(keys[x]))
			file_performanceTotal.write("\n\nTotal Mean Abs Error " + str(totalAE[x] / len(modelstreams)))
			file_performanceTotal.write("\nTotal Mean Squared Error " + str(totalSE[x] / len(modelstreams)))
			file_performanceTotal.write("\nTotal Mean Root Squared Error " + str(totalMRSE[x] / len(modelstreams)) + "\n")

	file_performanceTotal.close()


def evaluateModel(model, performanceStream = None, modelName = ""):


    if performanceStream:
        performanceStream.write('---------------------------------------------------------------\n' + modelName + '\n+---------------------------------------------------------------\n')
        performanceStream.write('MAE: ' + str(model.meanAbsError) + "\t\t MSE: " + str(model.meanSquaredError) + "\t\t MRSE: " + str(model.meanRootSquaredError) + "\n")


def evaluateRuns():
    rootFolder = './ITMM_Evaluation/'


    filePath = "../Datasets/Heterogeneity/Constraints/rats_static_60.csv"
    filePath = "../Datasets/Heterogeneity/Deployment/rats_thick.csv"
    filePath = "../Datasets/Heterogeneity/Deployment/rats_thin.csv"
    filePath = "../Datasets/Heterogeneity/MAC/aloha.csv"
    filePath = "../Datasets/Heterogeneity/SamplingPeriods/rats_default_smip.csv"

    filePath = "../../Datasets/Dynamism/Network Size/dyn_ns_rats.csv"
    filePath = "../../Datasets/Dynamism/Payload Size/dyn_ps_rat.csv"
    filePath = "../../Datasets/Dynamism/Sampling Period/dyn_sp_inc.csv"
    filePath = "../../Datasets/Dynamism/Spreading Factor/dyn_sf_rat.csv"
    filePath = "../../Datasets/Dynamism/Sampling Period/dyn_sp_dec.csv"


    # inc removed because of small size
    csv_files = {
        "dyn_sFac": "../../Datasets/Dynamism/Spreading Factor/dyn_sf_rat.csv",
        "payloadSize": "../../Datasets/Dynamism/Payload Size/dyn_ps_rat.csv",
        'constraints': "../../Datasets/Heterogeneity/Constraints/rats_static_60.csv",
        'deploymentThin': "../../Datasets/Heterogeneity/Deployment/rats_thin.csv",
        'deploymentThick': "../../Datasets/Heterogeneity/Deployment/rats_thick.csv",
        "networkSize": "../../Datasets/Dynamism/Network Size/dyn_ns_rats.csv",
        "dyn_sp_inc": "../../Datasets/Dynamism/Sampling Period/dyn_sp_inc.csv",
        "dyn_sp_dec": "../../Datasets/Dynamism/Sampling Period/dyn_sp_dec.csv",
        "macTSCH": "../../Datasets/Heterogeneity/MAC/tsch.csv",
        'static': "../../Datasets/rats_static.csv",
        "sampling": "../../Datasets/Heterogeneity/SamplingPeriods/rats_default_smip.csv"}



    ################################################################
    models = ["ARMAX_K"]
    ################################################################

    directory = os.path.dirname(rootFolder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("Evaluation-%d-%m-%Y_%H-%M-%S/")

    timeRoot = rootFolder + dt_string
    directory = os.path.dirname(timeRoot)
    if not os.path.exists(directory):
        os.makedirs(directory)

    csvdata = timeRoot+"data.csv"
    append_list_as_row(csvdata,["scenario","stream","constraint","mae","rmse","mer","altAR", "altModel", "altCom"])

    for file in csv_files.keys():

        streams = dict()
        for m in models:
            streams[m] = []

        streamRoot = timeRoot + file + '/'
        directory = os.path.dirname(streamRoot)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = dict()

        if file == "dyn_sp_dec":
            with open(csv_files[file], 'r', encoding='utf-8-sig') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                line_count = 0
                print(csv_reader.fieldnames)
                for row in csv_reader:
                    print(row['Series'][-4:], " ", row["peripheral"].replace("/", "_"), " ", row['value'])
                    key = row['Series'][-4:] + "_" + row["peripheral"].replace("/", "_")
                    if key not in data:
                        data[key] = []
                    else:
                        data[key].append(float(row['value']))
        else:
            with open(csv_files[file], 'r', encoding='utf-8-sig') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=';')
                line_count = 0
                for row in csv_reader:
                    line = str(row['Series'])
                    linesplit = line.split(' ')[2:]
                    key = ''
                    if len(linesplit) == 1:
                        key = linesplit[0][:-1]
                    else:
                        key = linesplit[0][:-1] + "_" + linesplit[2]
                        key = key[21:]
                    key = key.replace("/", '_').replace(",",'')
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


        file_performanceDump = open(streamRoot + 'output.txt', 'w')


        for key in datakeys:
            set = data[key]

            print(key)
            file_performanceDump.write(key)

            streamname = key
            streamNameRoot = streamRoot + streamname + '/'
            directory = os.path.dirname(streamNameRoot)
            if not os.path.exists(directory):
                os.makedirs(directory)

            for modelname in models:

                model = -1

                modelRoot = streamNameRoot + modelname + '/'
                directory = os.path.dirname(modelRoot)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                file_performance = open(modelRoot + 'performance.txt', 'w')
                file_performance.write(key + "\n")

                if modelname == 'ARMAX_K':
                    model = ARMAXKhronos_K(key,set, 10, modelRoot)
                    model.setKeys(keys)
                    model.filewriter = file_performance
                    model.dump = file_performanceDump
                    model.savePlots = True
                    model.makePlots = True
                    cnt = []
                    if file == 'dyn_sFac':
                        model.setScenario(parseCNT('sf')[key],parseSensorData('sf')[key])
                        model.sname = 'Spreading factor'
                        cnt = parseCNT('sf')[key]
                    if file == 'payloadSize':
                        model.setScenario(parseCNT('ps')[key], parseSensorData('ps')[key])
                        model.sname = 'Payload size'
                        cnt = parseCNT('ps')[key]
                    if file == 'networkSize':
                        model.scenario = parseNS()
                        model.sname = 'Network size'
                    if 'dec' in file or 'inc' in file:
                        model.scenario = set
                        model.sname = 'Sampling period'

                else:
                    print("Cannot run the given model: ", model)
                    continue

                with warnings.catch_warnings():
                	warnings.filterwarnings("ignore")
                	model.runSimulation()
                k = Khronos(key,set)
                if cnt != []:
                    k.setScenario(cnt,[])
                #k = ARMAXKhronos_K_es(key,set, 10, modelRoot)
                k.setKeys(keys)
                k.savePlots = False
                k.makePlots = False
                k.runSimulation()
                #model.runSimulation()
                file_performance.close()

                if model.scenario == []:
                    plotTwoModelRun(model,k,modelRoot)
                else:
                    plotTwomodelScenarioRun(model,k,modelRoot)
                plotConstraints(model,k,modelRoot)

                addData(csvdata, model, k, file, key)
                streams[modelname].append(model)
        file_performanceTotal = open(streamRoot + 'performance.txt', 'w')
        file_performanceTotal.write('PREDICTION TIME WINDOW ESTIMATIONS ' + str(len(streams[models[0]])))
        print(streams)
        writetotalinfo(models,streams,file_performanceTotal)





evaluateRuns()
# singleRun()
