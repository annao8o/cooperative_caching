import math
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(x, y, time_steps):
    print("-------create_dataset-------")
    #print("x.shape: ", x.shape, "y.shape: ", y.shape)

    Xs, Ys = [], []
    for i in range(len(x) - time_steps*2):
        v = x[i:(i + time_steps)]
        w = y[(i+time_steps): (i+(time_steps*2))]
        Xs.append(v)
        Ys.append(w)


    return np.array(Xs), np.array(Ys)



def getTraindata(train, x, timestamp):
    # split into train and test sets
    print("-------getTraindata-------")
    print(train.shape)
    train_x ,train_y = np.empty((0,timestamp)), np.empty((0,timestamp))

    for i in range(0,x):# content 수만큼 반복
        tmp_x, tmp_y = create_dataset(train[i,:], train[i,:], timestamp)

        #print(i,"번째,tmp_x.shape: ", tmp_x.shape,"tmp_y.shape: " ,tmp_y.shape)

        train_x= np.append(train_x,tmp_x,axis=0)
        train_y = np.append(train_y, tmp_y, axis=0)


    print("train_x.shape", train_x.shape)
    print("train_y.shape",train_y.shape)

    return train_x, train_y 
