# -*- coding: utf-8 -*-
from __future__ import division
import os,sys
import sqlite3
import MySQLdb as mdb

from data_model import *


class DataHandler():
	def __init__(self):
		self.conn = mdb.connect('221.147.129.137', 'james', 'mhrinc01', 'stock',port=63306, charset='utf8');
		#self.conn = mdb.connect('localhost', 'root', 'mhrinc01', 'race', port=63306, charset='utf8');
		cursor = self.conn.cursor()
		#cursor.execute("pragma journal_mode = MEMORY;")
		#cursor.execute("pragma journal_mode = WAL;")

	def beginTrans(self):
		self.conn.autocommit(False)

	def endTrans(self):
		try:
			self.conn.commit()
			self.conn.autocommit(True)
		except:
			print "Fatal Error in commit !!!"
			self.conn.rollback()


	def openSql(self,sql):
		try:
			cursor = self.conn.cursor()
			cursor.execute(sql)

			return cursor
			#gCursor.commit()
		except:
			#print "Unexpected error in ExecSQL:", sys.args[0]
			raise

	def execSql(self,sql,db_commit=True):
		try:
			cursor = self.conn.cursor()
			cursor.execute(sql)
			if db_commit:
				self.conn.commit()
			#cursor.close()
			return cursor

		except Exception as error:
			print ">>> Unexpected error in ExecSQL: ", error
			print "--- %s ---" % (sql)
			#raise


#print "dbhandler.py : cwd=%s" % (os.getcwd())
#data_handler = MyDataHandler()