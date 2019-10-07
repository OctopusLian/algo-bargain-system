# -*- coding: utf-8 -*-
from __future__ import division

import os,sys,datetime

from services import *
from charter import *
from data_model import *
from data_handler import *
from data_writer import *
from data_reader import *
from mess_trader import *
from data_crawler import *

from portfolio_builder import *
from predictor import *
from backtester import *
from alpha_model import *


if __name__ == "__main__":
	services.register('dbhandler',DataHandler())
	services.register('dbwriter',DataWriter())
	services.register('dbreader',DataReader())
	services.register('charter',Charter())
	services.register('configurator',Configurator())

	services.register('predictor',Predictors())
	services.register('trader',MessTrader())
	services.register('mean_reversion_model',MeanReversionModel())
	services.register('machine_learning_model',MachineLearningModel())


	crawler = DataCrawler()
	universe = Portfolio()
	portfolio = PortfolioBuilder()
	mean_backtester = MeanReversionBackTester()
	machine_backtester = MachineLearningBackTester()

	#crawler.updateAllCodes()
	#crawler.updateAllStockData(1,2010,1,1,2015,12,1,start_index=90)

	services.get('configurator').register('start_date','20150101')
	services.get('configurator').register('end_date','20151030')
	services.get('configurator').register('input_column','price_adj_close')
	services.get('configurator').register('output_column','indicator')
	services.get('configurator').register('data_limit',20)

	#finder.setTimePeriod('20150101','20151130')
	df_stationarity = portfolio.doStationarityTest('price_close')	
	df_rank = portfolio.rankStationarity(df_stationarity)
	stationarity_codes = portfolio.buildUniverse(df_rank,'rank',0.8)
	#print stationarity_codes

	
	df_machine_result = portfolio.doMachineLearningTest( split_ratio=0.75,lags_count=5 )
	df_machine_rank = portfolio.rankMachineLearning(df_machine_result)
	machine_codes = portfolio.buildUniverse(df_machine_rank,'rank',0.8)

	#print services.get('predictor').dump()
	#print df_machine_rank
	#print machine_codes
	
	universe.clear()
	universe.makeUniverse('price_close','stationarity',stationarity_codes)
	universe.makeUniverse('price_close','machine_learning',machine_codes)
	universe.dump()


	#machine_backtester.getConfusionMatrix('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.printClassificationReport('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.showROC('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.showROC('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.getHitRatio('rf','006650','20151101','20151130',lags_count=1)
	#machine_backtester.getHitRatio('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.getHitRatio('rf','006650','20151101','20151130',lags_count=10)
	#machine_backtester.drawHitRatio('rf','006650','20151101','20151130',lags_count=5)
	#machine_backtester.optimizeHyperparameter('rf','006650','20150101','20151130',lags_count=5)
	machine_backtester.optimizeHyperparameterByRandomSearch('rf','006650','20150101','20151130',lags_count=5)

	"""
	mean_backtester.setThreshold(1.5)
	mean_backtester.setWindowSize(20)
	
	#mean_backtester.doTest('stationarity',universe,'20150101','20151130')

	"""
	"""
	services.get('trader').setPortfolio(universe)
	services.get('trader').simulate()
	"""
	
	#services.get('trader').dump()


	#services.get('charter').drawStationarityTestHistogram(df)
	#services.get('charter').drawStationarityTestBoxPlot(df)
	#services.get('charter').drawStationarityRankHistogram(df_rank)

	#print df_rank
	#print codes