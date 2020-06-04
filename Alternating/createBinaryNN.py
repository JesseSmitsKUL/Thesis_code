from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout,TimeDistributed, GRU
from keras.callbacks import EarlyStopping
from sklearn.svm import SVC



# define model
EPOCHS = 350
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.4
DROPOUT_RATE_1 = 0.4
DROPOUT_RATE_2 = 0.3
DROPOUT_RATE_3 = 0.1
DROPOUT_RATE_4 = 0.5
DROPOUT_RATE_5 = 0.2
LATENT_DIM = 50

#define callbacks
MIN_DELTA = 0.01
PATIENCE = 50
es = EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA, patience=PATIENCE,verbose=1, restore_best_weights=True)


# saving state
saving = True


def transformAltStream(stream,length):
    x_values = []
    y_values = []
    for x in range(length,len(stream)-1):
        x_val = stream[x-length:x]
        x_last = x_val[-1]
        y_val = stream[x]
        if x_last > y_val:
            y_values.append(0)
        else:
            y_values.append(1)
        x_values.append(x_val)
    return (x_values,y_values)


def createSVM(x,y):
    clf = SVC(C=5)
    clf.fit(x,y)

    return (clf,'')



def createAltModelLSTM(x,y):
    X = array(x)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # define model
    model = Sequential()
    #model.add(GRU(50, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(LSTM(50, activation='relu', input_shape=(len(x[1]), 1)))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(X, array(y), epochs=EPOCHS, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, callbacks=[es])

    model.save('alt_model.h5')

    return (model, history)