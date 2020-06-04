import sys
sys.path.append("..")

import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
from Khronos import Khronos
from Evaluate_KValues.ARMAXKhronos_K import ARMAXKhronos_K
import copy as cp

from PySide2.QtWidgets import (QApplication)

from qt_vision.testing import MainWindow

import threading
import time
import random

import warnings

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


def createmodels(path, file, streamindex):

	CUTOFF = 0.7

	makePlots = False
	savePlots = False
	timeWindows = True
	saveLocation = "./Visuals/ExpSmoothing/"
	type = "static"

	fullpath = path + file




	data = dict()

	if file == "dyn_sp_dec":
		with open(fullpath, 'r', encoding='utf-8-sig') as csvfile:
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
		with open(fullpath, 'r', encoding='utf-8-sig') as csvfile:
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
				key = key.replace("/", '_')
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

	key = datakeys[streamindex]

	stream = data[key]

	cutoff = round(len(stream) * CUTOFF)

	print('\t' + key)
	maxlen = min(10000, len(stream))

	keys = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]    #0.3,0.5,0.6,0.7,0.8,0.9]

	keystr = [str(k) for k in keys]

	print(len(stream))

	#streamModel = NNKhronos(key,stream,'cnn_gru_test',CUTOFF, type)
	#streamModel = ARMAXKhronos(key,stream,10,type)
	streamModel = ARMAXKhronos_K(key,stream, 10, type)
	streamModel.setKeys(keys)
	streamModel.setVisualConfiguration(makePlots)
	streamModelKhronos = Khronos(key,stream)
	streamModelKhronos.setKeys(keys)
	streamModelKhronos.setVisualConfiguration(makePlots)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")

	return streamModel, streamModelKhronos, keystr


def runModel(model,t):

	for x in range(2000):
		# print("runmodel")
		model.updateArrival()
		time.sleep(t)

def threader(window, armax, khronos):

	time.sleep(10)

	while armax.index == -1:
		print("not loaded")
		time.sleep(1)

	print("update times")
	for x in range(10000):
		key = window.cb.currentText()

		size = len(armax.predictedArrivals)
		key = float(key)


		khronosAr = khronos.timeouts[key]
		khronosVio = khronos.constraint_violations[key]
		khronospred = khronos.predictedArrivals

		armaxAr = armax.timeouts[key]
		armaxVio = armax.constraint_violations[key]
		armaxpred = armax.predictedArrivals

		window.canvas[0].ydata = cp.deepcopy(khronosAr) if len(khronosAr) == len(khronospred) else cp.deepcopy(khronosAr[:-1])
		window.canvas[1].ydata = khronosVio
		window.canvas[2].ydata = cp.deepcopy(armaxAr) if len(armaxAr) == len(armaxpred) else cp.deepcopy(armaxAr[:-1])
		window.canvas[3].ydata = armaxVio


		window.canvas[0].arrivals = cp.deepcopy(khronospred)
		window.canvas[2].arrivals = cp.deepcopy(armaxpred)

		print("update")
		print("Ar: ", len(window.canvas[2].ydata), " ", len(armaxpred) )
		print("Khr: ", len(window.canvas[0].ydata), " ", len(khronospred) )

		time.sleep(0.5)


#     for x in range(100):
#         n = random.randint(0, 100)
#         window.canvas[0].ydata.append(n)
#         window.canvas[1].ydata.append(n*2)
#         window.canvas[2].ydata.append(n/2)
#         window.canvas[3].ydata.append(n*4)
#         time.sleep(2)
#






if __name__ == "__main__":

	filePath = "../../Datasets/Heterogeneity/Constraints/"
	file = "rats_static_60.csv"

	armax, khronos, keys = createmodels(filePath,file,0)

	app = QApplication(sys.argv)
	w = MainWindow(keys)


	t = threading.Thread(target=threader, args=(w,armax,khronos,))
	t.start()

	armaxThread = threading.Thread(target=runModel, args=(armax,0.01))
	khronosThread = threading.Thread(target=runModel, args=(khronos,0.2))

	# armax.runSimulation()

	#khronos.runSimulation()

	armaxThread.start()
	khronosThread.start()

	app.exec_()
