import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense,LSTM,Dropout,Activation
import datetime

class LSTM_MODEL:
    def __init__(self, model_name , model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.window_len = 5
        self.from_date = None
        self.to_date = None
        self.input_data = None

    def get_period_time(self):
        print("get_period_time")
        today = datetime.datetime.now()
        d = datetime.timedelta(days = self.window_len - 2)
        self.from_date = datetime.datetime.strftime(today - d, r"%Y-%m-%d" )
        self.to_date = datetime.datetime.strftime(today ,r"%Y-%m-%d" )
        
    def get_yahoo_data(self,all = True):
        print("get_yahoo_data")
        self.get_period_time()
        if "BTC" in self.model_name:
            coin_name = "BTC-USD"    
        if "ETH" in self.model_name:
            coin_name = "ETH-USD"
        df = web.DataReader(coin_name, data_source = 'yahoo', start = self.from_date, end = self.to_date)
        #df.to_csv("test.csv")
        self.input_data = df
        return df 

    def get_lstm_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.window_len,1)))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.load_weights(self.model_path)
        print("model loaded")
        self.model = model

    def predict_result(self):
        scaler = MinMaxScaler(feature_range = (0,1))
        target_data_column = self.input_data.filter(['Close'])
        scaled_input_data = scaler.fit_transform(target_data_column)
        scaled_input_data = np.array(scaled_input_data)
        scaled_input_data = np.reshape(scaled_input_data , (1, self.window_len, 1))
        preds = self.model.predict(scaled_input_data)
        preds = scaler.inverse_transform(preds)
        return preds



# model_btc = LSTM_MODEL("BTC","models/lstm_model_2.h5")
# data = model_btc.get_yahoo_data()
# print(data)
#model_btc.get_lstm_model()
#model_btc.predict_result()