#-*- coding: utf-8 -*-
import datetime

import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

def download_stock_data(file_name,company_code,year1,month1,date1,year2,month2,date2):
	start = datetime.datetime(year1, month1, date1)
	end = datetime.datetime(year2, month2, date2)
	df = web.DataReader("%s.KS" % (company_code), "yahoo", start, end)

	df.to_pickle(file_name)

	return df

# file_name:指定保存已下载股价数据的文件名
# company_code:指定股票代码
# year1,month1,date1:按照年、月、日的顺序输入下载数据的开始日期
# year2,month2,date2:按照年、月、日的顺序输入下载数据的结束日期


def load_stock_data(file_name):
	df = pd.read_pickle(file_name)
	return df

#download_stock_data('lg.data','066570',2015,1,1,2015,11,30)
#download_stock_data('samsung_2010.data','005930',2010,1,1,2015,11,30)
#download_stock_data('hanmi.data','128940',2015,1,1,2015,11,30)

df = load_stock_data('samsung_2010.data')
df['Open'].plot()
plt.axhline(df['Open'].mean(),color='red')
plt.show()
#print df

print df.describe()

"""

#print df.quantile([.25,.5,.75,1])

#(n, bins, patched) = plt.hist(df['Open'])
#df['Open'].plot(kind='kde')
#plt.axvline(df['Open'].mean(),color='red')
#plt.show()

for index in range(len(n)):
	print "Bin : %0.f, Frequency = %0.f" % (bins[index],n[index])
"""

#scatter_matrix(df[['Open','High','Low','Close']], alpha=0.2, figsize=(6, 6), diagonal='kde')
#df[['Open','High','Low','Close','Adj Close']].plot(kind='box')
#plt.show()