#-*- coding: utf-8 -*-
import os,datetime

import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


def get_data_file_path(file_name):
	full_file_name = "%s/data/%s" % (os.path.dirname(os.path.abspath(__file__)),file_name)
	#print full_file_name
	return full_file_name

def save_stock_data(df,file_name):
	new_file_name = get_data_file_path(file_name)
	df.to_pickle(new_file_name)

def download_stock_data(file_name,company_code,year1,month1,date1,year2,month2,date2):
	start = datetime.datetime(year1, month1, date1)
	end = datetime.datetime(year2, month2, date2)
	df = web.DataReader("%s.KS" % (company_code), "yahoo", start, end)
	save_stock_data(df,file_name)

	return df

def load_stock_data(file_name):
	new_file_name = get_data_file_path(file_name)
	df = pd.read_pickle(new_file_name)

	return df


