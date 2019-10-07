# -*- coding: utf-8 -*-
from __future__ import division

import os,sys,datetime,pickle
import requests, json
import BeautifulSoup



class BaseCollection:
    def __init__(self):
        self.items = {}

    def clear(self):
        self.items.clear()


    def count(self):
        return len(self.items.key())

    def countItem(self,column):
        return len(self.items[column])


    def find(self,column):
        if self.items.has_key(column):
            return self.items[column]
        return None


    def iterItems(self):
        return self.items.iterItems()



class StockCodeItem:
    def __init__(self,market_type,code,full_code,company):
        self.market_type = market_type
        self.code = code
        self.full_code = full_code
        self.company = company


class StockCode:
    def __init__(self):
        self.items = {}

    def count(self):
        return len(self.items)

    def clear(self):
        self.items.clear()

    def add(self,market_type,code,full_code,company):
        a_item = StockCodeItem(market_type,code,full_code,company)
        self.items[code] = a_item

    def remove(self,stock_code):
        del self.items[stock_code]

    def find(self,stock_code):
        return self.items[stock_code]

    def iterItems(self):
        return self.items.iteritems()

    def dump(self):
        index = 0
        for key,value in self.items.iteritems():
            print "%s : %s - Code=%s, Full Code=%s, Company=%s" % (index, value.market_type, key, value.full_code, value.company)
            index += 1



class PortfolioItem:
    def __init__(self,index,column,code,company):
        self.index = index
        self.column = column
        self.code = code
        self.company = company
        self.df = None
        self.score = 0

    def setData(self,df):
        self.df = df


class Portfolio(BaseCollection):
    def findCode(self,model,code):
        model = self.find(model)
        if model is None:
            return None

        for a_item in self.items[model]:
            if a_item.code == code:
                return a_item

        return None


    def add(self,column,model,code,company):
        portfolio = self.find(model)
        if portfolio is None:
            self.items[model] = []

        #if self.findCode(model,code) is not None:
        #    return 

        a_item = PortfolioItem(self.countItem(model),column,code,company)

        self.items[model].append( a_item )


    def makeUniverse(self,column,model,stock_dict):
        for key in stock_dict.keys():
            self.add(column,model,key,stock_dict[key])


    def dump(self):
        print ">>> Portfolio.dump <<<"
        for key in self.items.keys():
            print "- model=%s" % (key)
            for a_item in self.items[key]:
                print "... column=%s : index=%s, code=%s, company=%s" % (a_item.column,a_item.index,a_item.code,a_item.company)

        print "--- Done ---"


class TradeItem:
    def __init__(self,model,code,row_index,position):
        self.model = model
        self.code = code
        #self.trade_date = trade_date
        self.row_index = row_index
        self.position = position

