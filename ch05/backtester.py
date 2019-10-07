# -*- coding: utf-8 -*-
from __future__ import division
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint

from data_handler import *
from data_model import *
from services import *

from const import *
from stock_common import *



class BaseBackTester():
	def __init__(self):
		self.dbhandler = services.get('dbhandler')
		self.dbreader = services.get('dbreader')
		self.trader = services.get('trader')
		self.config = services.get('configurator')

		self.model = None
		self.start_date = None
		self.end_date = None


	def setTimePeriod(self,start,end):
		self.start_date = convertStringToDate(start)
		self.end_date = convertStringToDate(end)


	def loadDataFrames(self,model,portfolio,start_date,end_date):
		for a_item in portfolio.items[model]:
			a_item.df = self.dbreader.loadDataFrame(a_item.code,start_date,end_date)


	def doTest(self,portfolio):
		pass


	def getHitRatio(self):
		pass



class MeanReversionBackTester(BaseBackTester):
	def __init__(self):
		BaseBackTester.__init__(self)
		self.model = services.get('mean_reversion_model')

	def setWindowSize(self,size):
		self.window_size = size

	def setThreshold(self,threshold):
		self.threshold = threshold


	def doTest(self,model,portfolio,start_date,end_date):
		self.loadDataFrames(model,portfolio,start_date,end_date)
		
		for a_item in portfolio.items[model]:
			for row_index in range(a_item.df.shape[0]):
				if (row_index+1)>self.window_size:
					position = self.determinePosition(a_item.df,a_item.column,row_index)
					if position!=HOLD:
						self.trader.add(model,a_item.code,row_index,position)

		#self.trader.dump()

	def getHitRatio(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ]


		self.loadDataFrames(model,portfolio,start_date,end_date)
		
		for a_item in portfolio.items[model]:
			for row_index in range(a_item.df.shape[0]):
				if (row_index+1)>self.window_size:
					position = self.determinePosition(a_item.df,a_item.column,row_index)
					if position!=HOLD:
						self.trader.add(model,a_item.code,row_index,position)


		pred = classifier.predict(x_test)

		hit_count = 0
		total_count = len(y_test)
		for index in range(total_count):
			if (pred[index]) == (y_test[index]):
				hit_count = hit_count + 1

		hit_ratio = hit_count/total_count
		score = classifier.score(x_test, y_test)
		#print "hit_count=%s, total=%s, hit_ratio = %s" % (hit_count,total_count,hit_ratio)

		return hit_ratio, score
		# Output the hit-rate and the confusion matrix for each model

		#print("%s\n" % confusion_matrix(pred, y_test))



class MachineLearningBackTester(BaseBackTester):
	def __init__(self):
		BaseBackTester.__init__(self)
		self.model = services.get('machine_learning_model')
		self.predictor = services.get('predictor')


	def getTestDataset(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ]
		
		df_y_pred,df_y_pred_proba = a_predictor.predict(df_x_test.values)

		return df_x_test,df_y_true,df_y_pred


	def showROC(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)
		df_x_test,df_y_true,df_y_pred = self.getTestDataset(name, code, start_date,end_date,lags_count)
		a_predictor.drawROC(df_y_true,df_y_pred)
		

	def getConfusionMatrix(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ]
		
		#print df_x_test
		#print df_y_true.values

		df_y_pred,df_y_pred_proba = a_predictor.predict(df_x_test.values)

		#print df_y_pred

		print a_predictor.confusionMatrix(df_y_true,df_y_pred)
		#print pd.crosstab(df_y_true,df_y_pred)


	def printClassificationReport(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ]
		
		#print df_x_test
		print df_y_true.values

		df_y_pred,df_y_pred_proba = a_predictor.predict(df_x_test.values)

		print df_y_pred

		print a_predictor.classificationReport(df_y_true,df_y_pred,['Down','Up'])
		#print pd.crosstab(df_y_true,df_y_pred)


	def getHitRatio(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ].values

		"""
		self.loadDataFrames(model,portfolio,start_date,end_date)
		
		for a_item in portfolio.items[model]:
			for row_index in range(a_item.df.shape[0]):
				if (row_index+1)>self.window_size:
					position = self.determinePosition(a_item.df,a_item.column,row_index)
					if position!=HOLD:
						self.trader.add(model,a_item.code,row_index,position)
		"""

		df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)

		#print df_y_pred[0]

		ax = df_dataset[ [self.config.get('input_column')] ].plot()

		hit_count = 0
		total_count = len(df_y_true)
		for row_index in range(total_count):
			if (df_y_pred[row_index] == df_y_true[row_index]):
				hit_count = hit_count + 1
				ax.annotate('Yes', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

		hit_ratio = hit_count/total_count
		#score = classifier.score(x_test, y_test)
		print "hit_count=%s, total=%s, hit_ratio = %s" % (hit_count,total_count,hit_ratio)

		plt.show()

		return hit_ratio
		# Output the hit-rate and the confusion matrix for each model

		#print("%s\n" % confusion_matrix(pred, y_test))


	def drawHitRatio(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ].values


		df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)


		ax = df_dataset[ [self.config.get('input_column')] ].plot()

		for row_index in range(df_y_true.shape[0]):
			if (df_y_pred[row_index] == df_y_true[row_index]):
				ax.annotate('Yes', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))


		plt.show()
		# Output the hit-rate and the confusion matrix for each model



	def drawDrawdown(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ].values


		df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)


		ax = df_dataset[ [self.config.get('input_column')] ].plot()

		for row_index in range(df_y_true.shape[0]):
				
			position = self.model.determinePosition(code,df_dataset,self.config.get('input_column'),row_index)
			if position==LONG:
				ax.annotate('Long', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,-30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
			elif position==SHORT:
				ax.annotate('Short', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

			if (df_y_pred[row_index] == df_y_true[row_index]):
				ax.annotate('Yes', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))


		plt.show()
		# Output the hit-rate and the confusion matrix for each model



	def drawPosition(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 
		df_x_test = df_dataset[ [self.config.get('input_column')] ]
		df_y_true = df_dataset[ [self.config.get('output_column')] ].values


		df_y_pred,df_y_pred_probability = a_predictor.predict(df_x_test)


		ax = df_dataset[ [self.config.get('input_column')] ].plot()

		for row_index in range(df_y_true.shape[0]):
			if (row_index+1)>lags_count:
				
				#determinePosition(self,code,df,column,row_index,verbose=False):

				position = self.model.determinePosition(code,df_dataset,self.config.get('input_column'),row_index)
				if position==LONG:
					ax.annotate('Long', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,-30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
				elif position==SHORT:
					ax.annotate('Short', xy=(row_index, df_dataset.loc[ row_index, self.config.get('input_column') ]), xytext=(10,30), textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))


		plt.show()
		# Output the hit-rate and the confusion matrix for each model		


	def optimizeHyperparameter(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 

		X_train,X_test,Y_train,Y_test = self.predictor.splitDataset(df_dataset,'price_date',[self.config.get('input_column')],self.config.get('output_column'),split_ratio=0.8)

		param_grid = {"max_depth": [3, None],
					"min_samples_split": [1, 3, 10],
					"min_samples_leaf": [1, 3, 10],
					"bootstrap": [True, False],
					"criterion": ["gini", "entropy"]}

		a_predictor.doGridSearch(X_train.values,Y_train.values,param_grid)


	def optimizeHyperparameterByRandomSearch(self,name, code, start_date,end_date,lags_count=5):
		a_predictor = self.predictor.get(code,name)

		df_dataset = self.predictor.makeLaggedDataset(code,start_date,end_date, self.config.get('input_column'), self.config.get('output_column'),lags_count ) 

		X_train,X_test,Y_train,Y_test = self.predictor.splitDataset(df_dataset,'price_date',[self.config.get('input_column')],self.config.get('output_column'),split_ratio=0.8)

		param_dist = {"max_depth": [3, None],
					"min_samples_split": sp_randint(1, 11),
					"min_samples_leaf": sp_randint(1, 11),
					"bootstrap": [True, False],
					"criterion": ["gini", "entropy"]}

		a_predictor.doRandomSearch(X_train.values,Y_train.values,param_dist,20)
		print sp_randint(1, 11)