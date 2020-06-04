from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM, Dropout,TimeDistributed, GRU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# define model
EPOCHS = 150
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.3
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3
DROPOUT_RATE_3 = 0.1
DROPOUT_RATE_4 = 0.5
DROPOUT_RATE_5 = 0.2
LATENT_DIM = 50

#define callbacks
MIN_DELTA = 0.01
PATIENCE = 20
es = EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=PATIENCE,verbose=1, restore_best_weights=True)


def transformStream(stream,length):
    x_values = []
    y_values = []
    for x in range(length,len(stream)-1):
        x_val = stream[x-length:x]
        y_val = stream[x]
        x_values.append(x_val)
        y_values.append(y_val)
    return (x_values,y_values)

def custom_loss(y_true, y_pred):
    print(type(y_true))
    print(type(y_pred))
    return K.mean(K.square(y_pred - y_true), axis=-1)


def createModelMLP(x,y):
    # define model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=len(x[1])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss=costum_loss)
    # fit model
    model.fit(array(x), array(y), epochs=100)
    return model


def createModelCNN(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)
    return (model, history)


def createModelCNN2(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)
    return (model, history)


def createModelLSTM(x,y):

    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(Dense(50, activation='linear'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])
    return (model, history)


def createModelGRU(x,y):

    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=(len(x[1]), 1))) #,return_sequences=True
    #model.add(GRU(50, activation='relu')) # , return_sequences=True
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])
    return (model, history)

def createModelGRU2(x,y):

    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    model.add(GRU(50, activation='relu',return_sequences=True, input_shape=(len(x[1]), 1)))
    model.add(GRU(50, activation='relu')) # , return_sequences=True
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])
    return (model, history)


def createModelCNN_LSTM(x,y):

    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(TimeDistributed(Flatten()))

    # define LSTM model
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])
    # return (model, history)

    #
    # X = array(x)
    # X = X.reshape((X.shape[0], X.shape[1], 1))
    # # define model
    # cnn = Sequential()
    # cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(len(x[1]), 1)))
    # cnn.add(MaxPooling1D(pool_size=2))
    # cnn.add(Flatten())
    #
    # model = Sequential()
    # model.add(TimeDistributed(cnn.layers[-1]))
    # model.add(LSTM(50, activation='relu', input_shape=(len(x[1]), 1)))
    # model.add(Dense(1, activation='linear'))
    #
    # model.compile(loss='mae', optimizer='adam')
    # # fit model
    # history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE)
    model.summary()
    return (model, history)

def createModelCNN_LSTM2(x,y):

    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(TimeDistributed(Flatten()))

    # define LSTM model
    model.add(LSTM(80, activation='relu', return_sequences=True))
    model.add(LSTM(80, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(1, activation='linear'))
    #model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])
    return (model, history)

def createModelCNN_GRU(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(TimeDistributed(Flatten()))

    # define GRU model
    #model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(50, activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1, activation='linear'))
    # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es], verbose=0) #edit
    return (model, history)

def createModelCNN_GRU2(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(TimeDistributed(Flatten()))

    # define GRU model
    model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(50, activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1, activation='linear'))
    # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es], verbose=0) #edit
    return (model, history)

def createModelCNN_GRU_TEST(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Flatten()))

    # define GRU model
    model.add(GRU(50, activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es], verbose=1) #edit
    return (model, history)
