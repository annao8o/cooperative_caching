# reference:
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import traindata
import math
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import categorical_crossentropy


def show_multi_plot(history,true_future ,model_prediction, title):

    labels = ["History", "True Future", "Model Prediction"]
    marker = ["-", "bo", "ro"]
    num_in = list(range(-(history.shape[0]), 0))
    num_out = len(model_prediction)

    pyplot.plot(num_in, history[:,], marker[0], label=labels[0])
    pyplot.plot(np.arange(num_out), np.array(true_future[:, ]), marker[1], label=labels[1])

    if model_prediction.any():
        pyplot.plot(np.arange(num_out), model_prediction[:, ], marker[2], label=labels[2])

    pyplot.title(title)

    pyplot.show()
    return


def show_plot2(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    pyplot.title(title)
    for i, val in enumerate(plot_data):
        if i:
            pyplot.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            pyplot.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    pyplot.legend()
    pyplot.xlim([time_steps[0], (future + 5) * 2])
    pyplot.xlabel("Time-Step")
    pyplot.show()
    return

def create_dataset(x, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(x) - time_steps):
        v = x[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def lstm_ex(data,x,y):
    # split into train and test sets
    hours=y
    values=data
    print(values.shape)
    timestamp=24
    n_test_hours = timestamp*2 #(24 시간 보고 그 다음 24 시간 맞추기)
    n_train_hours = hours - n_test_hours

    print("n_train_hours: ",n_train_hours)

    # normalize features
#    vmax, vmin=values.max(), values.min()
#    values =(values -vmin)/(vmax-vmin)
#    values=values.transpose()
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(values)
    values=values.transpose()

    train = values[ : , :n_train_hours]
    test = values[ : ,n_train_hours :]
    print("train.shape : ",train.shape)
    print("test.shape : ", test.shape)

    # split into input and outputs
    train_x, train_y = traindata.getTraindata(train, x, timestamp=timestamp)

    test_x, test_y = test[:, 0:24], test[:,24:48]#(24 시간 보고 그 다음 24 시간 맞추기)
    #test_x, test_y = traindata.getTraindata(test, x, timestamp=timestamp)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # reshape input to be 3D [samples, timestamps, features]
    train_x = train_x.reshape((train_x.shape[0], timestamp, 1))
    test_x = test_x.reshape((test_x.shape[0], timestamp, 1))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


    # design network
    model = Sequential()
    model.add(LSTM(1, input_shape=(timestamp, 1)))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(24))
    #model.add(Activation('tanh'))
    model.compile(loss='mse', optimizer='adagrad' ,metrics=['accuracy'] )
    model.summary()
    # fit network
    history = model.fit(train_x, train_y, epochs=100, batch_size=4,validation_split=0.04, verbose=2, shuffle=False)

    # plot history
    epochs=range(len(history.history['loss']))
    pyplot.plot(epochs, history.history['loss'], label='train')
    pyplot.plot(epochs, history.history['val_loss'], label='test')
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend()
    # pyplot.show()


    pyplot.plot(epochs, history.history['accuracy'], label='train')
    pyplot.plot(epochs, history.history['val_accuracy'], label='test')
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend()
    # pyplot.show()


    # make a prediction
    yhat = model.predict(test_x)
    inv_yhat=yhat
    inv_y = test_y
    

##    for i in range(0,x):
##        show_multi_plot(test_x[i,:],test_y[i, : ], yhat[i,:],"multi-step prediction")

    score = model.evaluate(test_x, test_y)
    print(score)


    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    return yhat
