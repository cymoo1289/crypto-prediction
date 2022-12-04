import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense,LSTM,Dropout,Activation
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import datetime

def get_yahoo_data(coin):
    tdy = datetime.datetime.now()
    df = web.DataReader(coin, data_source = 'yahoo', start ='2018-01-01',end = datetime.datetime.strftime(tdy,r"%Y-%m-%d" ))
    print(df)
    return df

def get_period(period_len):
    pass

def plot_figure(df):
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close Price in USD')
    plt.show()

def preprocess_data(df):
    #get close column only 
    target_data = df.filter(['Close'])
    dataset = target_data.values
    # 80% train 20% test
    training_data_len = math.ceil(len(dataset) * 0.8) #The Math.ceil() function always rounds up and returns the smaller integer greater than or equal to a given number.
    
    train_data = df[:training_data_len]
    test_data = df[training_data_len:]

    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data_scaled = scaled_data[0:training_data_len , :]
    #print(train_data)
    window_len = 5 # this is the length of xinput prediction 7 means 7 days
    x_train = []
    y_train = []
    
    for i in range(window_len,len(train_data_scaled)):
        x_train.append(train_data_scaled[i-window_len:i,0])
        y_train.append(train_data_scaled[i,0])

    # convert to numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)
    #lstm model need 3d array, so reshape here
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #number of data , window len, number of features

    
    test_data_scaled = scaled_data[training_data_len - window_len: , :]
    #print("test_data",test_data)
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(window_len,len(test_data_scaled)):
        x_test.append(test_data_scaled[i-window_len:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1], 1))

    return train_data, test_data, x_train, y_train, x_test , y_test, scaler




def build_lstm_model(x_train=None, y_train=None, TRAINING = True):
    if x_train is None or y_train is None:
        TRAINING = False
    print("#########################")
    print("model_1")
    model_1 = Sequential()
    model_1.add(LSTM(50, return_sequences = True , input_shape = (x_train.shape[1],1)))
    model_1.add(LSTM(50, return_sequences = False))
    model_1.add(Dense(25))
    model_1.add(Dense(1))
    model_1.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model_1_weight_path = "../models/lstm_model_1.h5"
    if TRAINING:
        model_1.fit(x_train, y_train, batch_size = 32, epochs = 5)
        model_1.save(model_1_weight_path)
        plt.plot(model_1.history.history['loss'],'r',linewidth=2, label='Training loss')
        #plt.plot(model.history.history['val_loss'], 'g',linewidth=2, label='Validation loss')
        plt.title('LSTM Neural Networks - BTC Model')
        plt.xlabel('Epochs numbers')
        plt.ylabel('MSE numbers')
        plt.show()
    else:
        model_1.load_weights(model_1_weight_path)
        print("model_1 loaded")
    # model 2 
    print("#########################")
    print("model_2")
    model_2 = Sequential()
    model_2.add(LSTM(50, input_shape=(x_train.shape[1],1)))
    model_2.add(Dropout(0.2))
    model_2.add(Dense(units=1))
    model_2.add(Activation('linear'))
    model_2.compile(loss='mean_squared_error', optimizer='adam')
    model_2_weight_path = "../models/lstm_model_2.h5"
    if TRAINING:
        model_2.fit(x_train, y_train, batch_size = 32, epochs = 10)
        model_2.save(model_2_weight_path)
        plt.plot(model_2.history.history['loss'],'r',linewidth=2, label='Training loss')
        #plt.plot(model.history.history['val_loss'], 'g',linewidth=2, label='Validation loss')
        plt.title('LSTM Neural Networks - BTC Model')
        plt.xlabel('Epochs numbers')
        plt.ylabel('MSE numbers')
        plt.show()

    else:
        model_2.load_weights(model_2_weight_path)
        print("model_2 loaded")
    return model_1 , model_2

def show_plot(train_data, test_data, y_preds):
    test_data['predictions'] = y_preds
    plt.figure(figsize=(16,8))
    plt.title("Model")
    plt.xlabel("Date")
    plt.ylabel("Close price USD")
    plt.plot(train_data['Close'])
    plt.plot(test_data[['Close','predictions']])
    plt.legend(['Train','Test','Preds'], loc = 'lower right')
    plt.show()

data = get_yahoo_data("BTC-USD")
data.to_csv("btc-usd.csv",index = True)
plot_figure(data)
train_data, test_data, x_train, y_train,x_test , y_test , scaler = preprocess_data(data)
model_1, model_2 = build_lstm_model(x_train, y_train, True)
print(x_train)
print(y_train)
y_preds_1 = model_1.predict(x_test)
y_preds_2 = model_2.predict(x_test)

y_preds_1 = scaler.inverse_transform(y_preds_1)
y_preds_2 = scaler.inverse_transform(y_preds_2)

for i, preds in enumerate([y_preds_1,y_preds_2]):
    df = pd.DataFrame(zip(y_test,preds),columns = ['targets','preds'])
    df.to_csv("result_{}.csv".format(i),index=False)


rmse_1 = np.sqrt(np.mean(y_preds_1 - y_test)**2)
rmse_2 = np.sqrt(np.mean(y_preds_2 - y_test)**2)
print(rmse_1, rmse_2)
show_plot(train_data , test_data, y_preds_1)
show_plot(train_data , test_data, y_preds_2)


