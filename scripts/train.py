import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import os
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto
tf.get_logger().setLevel('INFO')

if tf.test.gpu_device_name():
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu)

else:

   print("Please install GPU version of TF")

data = pd.read_csv("../data/only_btc.csv")
data = data[:20000]
data = data.set_index('timestamp')
data.index = pd.to_datetime(data.index,unit='ns')
aim = "Close"
# target_data = data[aim].to_numpy()



def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('BTC/USDT', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.savefig("prediction.jpg")
    plt.show()

def normalise_zero_base(continuous):
    return continuous / continuous.iloc[0] - 1

def normalise_min_max(continuous):
    return (continuous - continuous.min()) / (data.max() - continuous.min())

def extract_window_data(continuous, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(continuous) - window_len):
        tmp = continuous[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(continuous, aim, window_len=5, zero_base=True, test_size=0.2):
    len_train = int(len(continuous)*0.8)
    train_data = continuous.iloc[:len_train]
    test_data = continuous.iloc[len_train-1:]
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[aim][window_len:].values
    y_test = test_data[aim][window_len:].values
    if zero_base:
        y_train = y_train / train_data[aim][:-window_len].values - 1
        y_test = y_test / test_data[aim][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model
    
np.random.seed(245)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 100
epochs = 50
batch_size = 64
loss = 'mse'
dropout = 0.2
optimizer = 'adam'
train_data, test_data, X_train, X_test, y_train, y_test = prepare_data(
    data, aim, window_len=window_len, zero_base=zero_base, test_size=test_size)

print(len(X_train), len(X_test), len(y_train), len(y_test))

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
'''modelfit = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
model.save("../models/lstm.h5")
plt.plot(modelfit.history['loss'],'r',linewidth=2, label='Training loss')
plt.plot(modelfit.history['val_loss'], 'g',linewidth=2, label='Validation loss')
plt.title('LSTM Neural Networks - BTC Model')
plt.xlabel('Epochs numbers')
plt.ylabel('MSE numbers')
plt.show()'''
model.load_weights("../models/lstm.h5")
targets = test_data[aim][window_len:]
preds = model.predict(X_test).squeeze()
print(mean_absolute_error(preds, y_test))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
print(len(X_train), len(X_test), len(y_train), len(y_test))



# r2_score=r2_score(y_test, preds)
# print(r2_score*100)

preds = test_data[aim].values[:-window_len] * (preds + 1) #inverse normalize 
preds = pd.Series(index=targets.index, data=preds)


df = pd.DataFrame(zip(targets,preds),columns = ['targets','preds'])
print(df)
SCORE_MSE=mean_squared_error(df['targets'],df['preds'])
print(math.sqrt(SCORE_MSE))
df.to_csv("results.csv",index=False)
line_plot(targets[:100], preds[:100], 'actual', 'prediction', lw=3)

