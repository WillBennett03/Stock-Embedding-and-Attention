"""
###############
### ingestion.py ###
###############

~ Will Bennett 11/07/2021

this takes an index and gets the stock list from wikipedia and then gets the stock data from qunadl,
it returns values in a json form
"""
import pandas as pd
import pandas_datareader as web
import datetime
import numpy as np
import json

with open('config.json', 'r') as file:
    config = json.load(file)

def get_stock_comparative_data(stock_list, index_dict, max_length): #returns the rediculus datastructure of comparitive attention
    data = {}
    if stock_list == []:
        for i, symb in enumerate(index_dict):
            row = index_dict[symb]['data']
            zeros = np.zeros((max_length, 5)) #adds padding so not ragged tensor later.
            row_length = len(row)
            start_index = max_length - row_length
            zeros[:start_index] = row
            token_index = index_dict[symb]['stock_token'] #the stock token is in the form of a index to make a one hot token
            data[token_index] = zeros
    else:
        for symb in stock_list:
            row = index_dict[symb]['data']
            zeros = np.zeros((max_length, 5)) #adds padding so not ragged tensor later.
            row_length = len(row)
            start_index = max_length - row_length
            zeros[:start_index] = row
            token_index = index_dict[symb]['stock_token'] #the stock token is in the form of a index to make a one hot token
            data[token_index] = zeros
    return data



def get_index_table(index_name):
    r"""
        Args :
            index_name : String

        returns :
            pandas.Dataframe object 
        Takes the name of the index and searchs wikipedia and returns a pandas dataframe of its contences.
    """
    if index_name.upper() == 'SP500':
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    else:
        try:
            return pd.read_html('https://en.wikipedia.org/wiki/List_of_'+index_name+'_companies')
        except:
            print(" Wiki page not found, check you index name matches that of the URL of the page")
            return False

def get_stock_data(symbol):
    r"""
    Args :
        symbol : String
    returns:
        pandas.Dataframe object
    takes the stock symbol and requests quandl for the data and returns a pandas dataframe of it.
    """
    try:
        return web.DataReader(symbol, data_source='quandl', api_key='-QhYiYMYoTJ-_azWJ5Eh')
    except:                           
        return False

def get_dict(name):
        with open(name, 'r') as file:
            return json.load(file)

class index_data:
    def __init__(self, index_name, get_new=False, min_date_lim=None, max_date_lim=None, save=False):
        r"""
        Args : 
            index_table : string (name of the index found in the wiki url),
            get_new : Boolean, when get_new = True then a new json file is made.
            min_date_lim : float/None (the models minimum date if model has already been trained,)
            max_date_lim : float/None (the mdoels maximum date if model has already been trained.)
            save : Boolean (if True then will save file to preprocessed folder)
        Returns:
            Dict in the form of... 
            Index : {
    
                Symbol : {
                    sector : string,

                    index : int,

                    symbol_token: int,

                    min_date : float,

                    max_date: float,

                    time_token: int,

                    len : int,

                    data : 2d array, #it is also in the form of percent change and so normailsed
                },
                ...
            }
        """
        self.index_name = index_name
        self.min_date_lim = min_date_lim
        self.max_date_lim = max_date_lim
        self.save = save
        
        if get_new:
            self.index_dict = self.build_dict()
        else:
            self.index_dict = get_dict()
        
        self.preprocess_dict() 

    def build_dict(self):
        index_table = get_index_table(self.index_name)
        index_dict = {} 
        for i, stock_symbol in enumerate(index_table['Symbol']): #iterates over the stocks in the table
            stock_df = get_stock_data(stock_symbol) #gets tge stock data from quandl API
            if isinstance(stock_df, False):
                data, min_date, max_date = self.clean_stock_df(stock_df)
                
                if isinstance(min_date, self.min_date_lim) and isinstance(max_date, self.max_date_lim): #checks if lim is None
                    if min_date >= self.min_date_lim and max_date <= self.max_date_lim:#Checks if within range -In future add date selection
                        index_dict[stock_symbol] = {
                            'sector' : index_table['GICS Sector'][i],
                            'index' : i,
                            'stock_token' : i,
                            'length' : len(data),
                            'max_date' : max_date,
                            'min_date' : min_date,
                            'data' : data
                        }
                else:
                    index_dict[stock_symbol] = {
                            'sector' : index_table['GICS Sector'][i],
                            'index' : i,
                            'stock_token' : i,
                            'length' : len(data),
                            'max_date' : max_date,
                            'min_date' : min_date,
                            'data' : data
                        }

        if self.save:
            with open('data/'+self.index_name+'.json', 'w+') as file:
                json.dump(index_dict, file)
            print('saved to data/'+self.index_name+'.json!')

        self.index_dict = index_dict

    def clean_stock_df(self, stock_df):
        r"""
        Args:
            stock_df : pandas.Dataframe object
        returns:
            (2d list where all values in the form of % change, min_date, max_date)
        
        takes the raw Dataframe and converts it into a 2d list and takes the min and max date.
        """
        stock_df = stock_df.dropna()
        Dates = stock_df.index
        min_date = Dates[-1].date().isoformat() #df comes oriented in decending order so first index is largest and last is smallest
        max_date = Dates[0].date().isoformat()
        stock_df = stock_df.filter(['High', 'Low', 'Close', 'Open', 'Volume']) #filter
        stock_df = stock_df.pct_change() #scale
        data = stock_df.values.tolist()
        return data, min_date, max_date

    def preprocess_dict(self):
        r"""
        args:
            index_dict : Dict object
        returns:
            X : np.array,
            Y : np.array,
            Sector : np.array,
            Stock_vocab_size : int,
            Time_vocab_size : int,
        """
        
        if isinstance(self.max_date, None) and isinstance(self.min_date, None):
            self.get_min_max_date()

        self.time_vocab_size = (self.max_date-self.min_date).days
        self.stock_vocab_size = len(self.index_dict)
        print(f'Max date : {self.max_date} \n Min date : {self.min_date} \n {self.time_vocab_size}')

        self.get_normilised_data()

    def get_normilised_data(self):
        symb = []
        sect = []
        x = []
        y = []
        for i, stock in enumerate(self.index_dict):
            data = self.index_dict[stock]['data']
            symb.append(stock)
            sect.append(self.index_dict[stock]['sector'])
            current_min_date = datetime.datetime.strptime(self.index_dict[stock]['min_date'], '%Y-%m-%d')
            start_day = (current_min_date-self.min_date).days
            for j, row in enumerate(data):
                day = start_day + j + 1
                x.append([i,day])
                y.append(row)

        x = np.array(x)
        y = np.array(y)
        y = np.concatenate([x,y], axis=1) #concats so inf rows are treated as a table to keep data consistant
        y = y[~np.isposinf(y).any(axis=1)] #deletes row with inf values

        y = y.transpose()# transposes to X and Y can be extracted
        x = y[:2]#first two columns are X
        y = y[2:7]#last five columns are Y
        x = x.transpose()
        y = y.transpose()

        self.x = x
        self.y = y
        self.sect = sect
        self.symb = symb

    def get_min_max_date(self):
        self.min_date = datetime.datetime.now()
        self.max_date = datetime.datetime.now()

        for i, stock_symbol in enumerate(self.index_dict):
            current_max_date = datetime.datetime.strptime(self.index_dict[stock_symbol]['max_date'], '%Y-%m-%d')
            current_min_date = datetime.datetime.strptime(self.index_dict[stock_symbol]['min_date'], '%Y-%m-%d')
            if self.min_date > current_min_date:
                self.min_date = current_min_date
            if self.max_date < current_max_date:
                self.max_date = current_max_date


