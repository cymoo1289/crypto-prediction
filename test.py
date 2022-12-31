import datetime
import pandas_datareader as web

def get_yahoo_data(coin):
    tdy = datetime.datetime.now()
    df = web.DataReader(coin, data_source = 'yahoo', start ='2018-01-01',end = datetime.datetime.strftime(tdy,r"%Y-%m-%d" ))
    print(df)
    return df

get_yahoo_data("BTC-USD")