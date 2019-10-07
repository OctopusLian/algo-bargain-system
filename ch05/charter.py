#-*- coding: utf-8 -*-
import os,sys,datetime
import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from pandas.compat import range, lrange, lmap, map, zip
from pandas.tools.plotting import scatter_matrix,autocorrelation_plot

from services import *


class Charter:
	def __init__(self):
		pass


	def drawStationarityTestHistogram(self,df):
		fig, axs = plt.subplots(3,1)

		df['adf_5'].plot(kind='hist',title="ADF 5%",ax=axs[0])
		df['hurst'].plot(kind='hist',title="Hurst Exponent",ax=axs[1])
		df['halflife'].plot(kind='hist',title="Half Life",ax=axs[2])

		plt.show()


	def drawStationarityRankHistogram(self,df):
		fig, axs = plt.subplots(3,1)

		df['rank_adf'].plot(kind='hist',title="ADF",ax=axs[0])
		df['rank_hurst'].plot(kind='hist',title="Hurst Exponent",ax=axs[1])
		df['rank_halflife'].plot(kind='hist',title="Half Life",ax=axs[2])

		plt.show()


	def drawStationarityTestBoxPlot(self,df):
		df[ ['adf_5','hurst','halflife'] ].plot(kind='box',layout=(1,3),subplots=True,title="Stationarity Test",)
		plt.show()


	def plot_price_series(self,df, ts1, ts2):
	    months = mdates.MonthLocator()  # every month
	    fig, ax = plt.subplots()
	    ax.plot(df.index, df[ts1], label=ts1)
	    ax.plot(df.index, df[ts2], label=ts2)
	    ax.xaxis.set_major_locator(months)
	    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
	    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
	    ax.grid(True)
	    fig.autofmt_xdate()

	    plt.xlabel('Month/Year')
	    plt.ylabel('Price ($)')
	    plt.title('%s and %s Daily Prices' % (ts1, ts2))
	    plt.legend()
	    plt.show()


	def plot_scatter_series(self,df, ts1, ts2):
	    plt.xlabel('%s Price ($)' % ts1)
	    plt.ylabel('%s Price ($)' % ts2)
	    plt.title('%s and %s Price Scatterplot' % (ts1, ts2))
	    plt.scatter(df[ts1], df[ts2])
	    plt.show()


	def plot_residuals(self,df):
	    months = mdates.MonthLocator()  # every month
	    fig, ax = plt.subplots()
	    ax.plot(df.index, df["res"], label="Residuals")
	    ax.xaxis.set_major_locator(months)
	    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
	    ax.set_xlim(datetime.datetime(2012, 1, 1), datetime.datetime(2013, 1, 1))
	    ax.grid(True)
	    fig.autofmt_xdate()

	    plt.xlabel('Month/Year')
	    plt.ylabel('Price ($)')
	    plt.title('Residual Plot')
	    plt.legend()

	    plt.plot(df["res"])
	    plt.show()

