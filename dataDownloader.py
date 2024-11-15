import pandas as pd
import pandas_datareader as pdr
import requests
import yfinance as yf
from io import StringIO
import datetime
import numpy as np

class YahooFinance:   
    
    def __init__(self):
        
        self.data = pd.DataFrame()

    
    def getDailyData(self, marketSymbol, startingDate, endingDate):
        
        data = yf.download(marketSymbol,start=startingDate,end=endingDate)
        self.data = self.processDataframe(data)
        return self.data


    def processDataframe(self, dataframe):
        # Remove useless columns
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        return dataframe

    
class CSVHandler:
    
    
    def dataframeToCSV(self, name, dataframe):
        path = name + '.csv'
        dataframe.to_csv(path)
        
        
    def CSVToDataframe(self, name):
        
        path = name + '.csv'
        return pd.read_csv(path,
                           header=0,
                           index_col='Timestamp',
                           parse_dates=True)
    

    