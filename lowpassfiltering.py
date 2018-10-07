from __future__ import division
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import collections
from random import randint
from matplotlib import style
style.use('fivethirtyeight')
import pandas as pd
import numpy as np
from itertools import count
data = pd.read_csv('sunspots.txt', sep="\t", header=None, names = ["Months", "Sunspots"])

#Define some use-case specific User Defined Functions
def moving_average(data, window_size):
	window = np.ones(int (window_size))/float(window_size)
	return np.convolve(data, window, 'same')

def explain_anomalies(y, window_size, sigma=1.0):
	avg = moving_average(y, window_size).tolist()
	residual = y - avg

	#Calculate the variation in the distribution of the residual
	std = np.std(residual)
	return {'standard_deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict( [ (index, y_i) for index, y_i, avg_i in zip(count(), y, avg) if (y_i > avg_i + (sigma * std)) | (y_i < avg_i - (sigma * std))] )}

def explain_anomalies_rolling_std(y, window_size, sigma=1.0):
	avg = moving_average(y, window_size)
	avg_list = avg.tolist()
	residual = y - avg

	#Calculate the variation in the distribution of the residual
	testing_std = pd.rolling_std(residual, window_size)
	testing_std_as_df = pd.DataFrame(testing_std)
	rolling_std = testing_std_as_df.replace(np.nan, testing_std_as_df.ix[window_size - 1]).round(3).iloc[:, 0].tolist()

	std = np.std(residual)
	return {'stationary standard deviation': round(std, 3), 'anomalies_dict': collections.OrderedDict( [ (index, y_i) for index, y_i, avg_i, rs_i in zip(count(), y, avg_list, rolling_std) if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i)) ] )}

def plot_results(x, y, window_size, sigma_value=1, text_xlabel="X Axis", text_ylabel="Y Axis", applying_rolling_std=False):
	plt.figure(figsize=(15, 8))
	plt.plot(x, y, "k.")
	y_av = moving_average(y, window_size)
	plt.plot(x, y_av, color='green')
	plt.xlim(0, 1000)
	plt.xlabel(text_xlabel)
	plt.ylabel(text_ylabel)

	events = {}
	if applying_rolling_std:
		events = explain_anomalies_rolling_std(y, window_size=window_size, sigma=sigma_value)
	else:
		events = explain_anomalies(y, window_size=window_size, sigma=sigma_value)

	x_anomaly = np.fromiter(events['anomalies_dict'].keys(), dtype=int, count=len(events['anomalies_dict']))
	y_anomaly = np.fromiter(events['anomalies_dict'].values(), dtype=float, count=len(events['anomalies_dict']))

	plt.plot(x_anomaly, y_anomaly, "r*", markersize=12)

	plt.grid(True)
	plt.show()

x = data['Months']
Y = data['Sunspots']


plot_results(x, y = Y, window_size=10, text_xlabel="Months", sigma_value=3, text_ylabel="No. of Sun spots")
events = explain_anomalies(Y, window_size=5, sigma=3)

print("Information about the anomalies model:{}".format(events))
