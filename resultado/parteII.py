"""
Created on Sun Dec 20 16:37:18 2020
NN: RNN (LSTM) + CNN
Training, part II
@author: Santiago L. Zuñiga, santiago.zuniga@ib.edu.ar
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import models, layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import Callback
import pickle

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(911)

X = np.load('./X_train.npy')        #load data
y = np.load('./y_train.npy').astype('int')

#load scaler and label encoder
scalers = pickle.load(open('./scaler.pkl','rb'))
le = pickle.load(open('./labelencoder.pkl','rb'))

y2 = le.transform(y.flatten()) 

n_classes = y2.max()+1 #14

splitFlag = False #whether to split data into training and validation
if splitFlag:
    X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.10, shuffle = True, random_state = 911)
else:  
    X_train = X
    y_train = y2
    X_test = X_train[:100,:,:]
    y_test = y_train[:100]


#scale data using a standard scalar (ie, substract mean and divide by std)
for i in range(X_train.shape[1]):
    X_train[:, i, :] = scalers[i].transform(X_train[:, i, :]) 

for i in range(X_test.shape[1]):
    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 



# %%
class MyMetrics(Callback):
    '''
    callback for balanced accuracy score
    '''
    def __init__(self, validation_data=(), training_data=()):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.X_tra, self.y_tra = training_data
        
    def on_train_begin(self, logs={}):
        self.bal_score = []
        self.bal_scoreTr = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_pred, axis = 1)
        score = balanced_accuracy_score(self.y_val, y_pred)
        self.bal_score.append(score)
        if not epoch%5:
          print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
        y_pred = self.model.predict(self.X_tra, verbose=0)
        y_pred = np.argmax(y_pred, axis = 1)
        score = balanced_accuracy_score(self.y_tra, y_pred)
        self.bal_scoreTr.append(score)
        



def createModelII( reg1=1e-3, reg2=1e-3, dp = 0.3 ,
                  learning_rate = 1e-4):

    tsteps = 48
    tf.keras.backend.clear_session()
    model = models.Sequential()

    model.add(layers.LSTM(32,  input_shape=[tsteps,6],
                          return_sequences=True,dropout = 0.2, use_bias=True))
    
    model.add(layers.TimeDistributed(layers.Dense(tsteps, activation='relu', use_bias=True)))
    
    model.add(layers.Reshape([tsteps,tsteps,1]))
    
    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))

    
    model.add(layers.Conv2D(32, (7, 7), padding="same", activation="relu"))

    model.add(layers.Flatten())
    
    
    model.add(layers.Dense(64,
                           kernel_regularizer=tf.keras.regularizers.l2(reg1) ))
    model.add(layers.ELU())
    

    model.add(layers.Dropout(dp))
    
    model.add(layers.Dense(14, activation='softmax',
                           kernel_regularizer=tf.keras.regularizers.l2(reg2)))
    
    
    lr_decayed_fn = (tf.keras.experimental.CosineDecayRestarts(learning_rate,10))
    opt = tf.keras.optimizers.Adam(learning_rate = lr_decayed_fn, amsgrad=True,
                                   beta_1 = 0.86, beta_2 = 0.98, epsilon = 1e-9)

    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, )
    model.compile( optimizer = opt,loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
    
    return model


# %%
model = createModelII(reg1=1e-1, reg2=5e-2, dp = 0.5 ,
                  learning_rate = 5e-5)

#load weights from part I
model.load_weights('./modelParteI.h5', by_name = True, skip_mismatch=True) 


skmetrics = MyMetrics(validation_data=(X_test, y_test), training_data=(X_train, y_train))

# train
history = model.fit(X_train, y_train,batch_size= 16,  epochs=500, verbose=1, 
                    validation_data = (X_test, y_test),
                    callbacks=[ skmetrics],
                    )

model.save('modelParteII.h5')