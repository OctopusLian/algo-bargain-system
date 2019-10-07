# -*- coding: utf-8 -*-
from __future__ import division
import os,sys
import sqlite3
import MySQLdb as mdb

from data_handler import *
from data_model import *
from services import *

from stock_common import *


class DataReader():
	def __init__(self):
		self.dbhandler = services.get('dbhandler')


	def loadDataFrame(self,code,start_date,end_date):
		#print "loadDataFrame : %s, %s" % (start_date,end_date)

		converted_start_date = convertStringToDate(start_date)
		converted_end_date = convertStringToDate(end_date)

		sql = "select * from prices"
		sql += " where code='%s'" % (code)
		sql += " and price_date between '%s' and '%s' " % (converted_start_date,converted_end_date)

		#print sql

 		df = pd.read_sql(sql,self.dbhandler.conn)

 		return df


	def loadCodes(self,limit=0):
		sql = "select code,company from codes where market_type=1"
		if limit>0:
			sql += " limit %s" % (limit)

		rows = self.dbhandler.openSql(sql).fetchall()

		return rows
