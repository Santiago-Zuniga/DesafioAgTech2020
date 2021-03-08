"""
Created on Sun Dec 20 16:37:18 2020
NN: RNN (LSTM) + CNN
Training, part III
@author: Santiago L. Zuñiga, santiago.zuniga@ib.edu.ar
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import models, layers
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.callbacks import Callback
import pickle
from sklearn.utils import class_weight
from swa.tfkeras import SWA
from sklearn.utils import resample

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

np.random.seed(911)

X = np.load('./X_trainAug.npy')        #load data
y = np.load('./y_trainAug.npy').astype('int')

#load scaler and label encoder
scalers = pickle.load(open('./scaler.pkl','rb'))
le = pickle.load(open('./labelencoder.pkl','rb'))

y = le.transform(y.flatten()) 

n_classes = y.max()+1 #14


#scale data using a standard scalar (ie, substract mean and divide by std)
for i in range(X.shape[1]):
    X[:, i, :] = scalers[i].transform(X[:, i, :]) 


#create class weights dict. Augmented data has THE SAME proportions as original training data
weight = class_weight.compute_class_weight('balanced', np.unique(y),y)
weight = {i : weight[i] for i in range(weight.size)}

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
        



def createModelIII( reg1=1e-3, reg2=1e-3, dp = 0.3 ,
                  learning_rate = 1e-4):

    tsteps = 48
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

def trainModel(X_train, y_train, X_test, y_test):

  model = createModelIII(reg1=1e-1, reg2=5e-2, dp = 0.5 ,
                  learning_rate = 1e-4)
  model.load_weights('modelParteII.h5')
  # define swa callback
  swa = SWA(start_epoch=80, 
            lr_schedule='cyclic', 
            swa_lr=1e-8,
            swa_lr2=1e-5,
            swa_freq=5,
            verbose=0)
  # train
  model.fit(X_train, y_train,batch_size= 8,  epochs=200, verbose=0, 
                      callbacks=[swa, ],
                      class_weight = weight)

  ypred = model.predict(X_test)
  ypred = np.argmax(ypred, axis = 1)
  test_acc = balanced_accuracy_score(y_test,ypred)
  return model, test_acc

scores, members = list(), list()
used = list()
for co in range(20): #20 splits
  # select indexes
  ix = [i for i in range(len(X))]
  train_ix = resample(ix, replace=True, n_samples=1000) #generate a new set with 1000 samples
  test_ix = [x for x in ix if x not in train_ix]
  print('Model {} of {}'.format(co+1, 20))
  print('Unique training data: {}, testing data: {}'.format(X.shape[0]-len(test_ix), len(test_ix)))
  # select data
  X_train, y_train = X[train_ix], y[train_ix]
  X_test, y_test = X[test_ix], y[test_ix]
  # train each model
  model, test_acc = trainModel(X_train, y_train, X_test, y_test)
  print('Test accuracy: {:3.3f}'.format(test_acc))

  scores.append(test_acc)
  members.append(model) #this list will hold all trained models
  used+=train_ix

test_ix = [x for x in ix if x not in used]
print('Unique training data used: {}, percentage: {:2.2f}%'.format((X.shape[0]-len(test_ix)),
                                                             100*(X.shape[0]-len(test_ix))/X.shape[0]))
print('Estimated Accuracy %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# %%
def defineStackedModel(members):
  '''
  create a single model
  '''
  # update all layers in all models to not be trainable
  ensemble_visible = []
  for i in range(len(members)):
    model = members[i]
    for layer in model.layers:
      # make not trainable
      layer.trainable = False
      # rename to avoid 'unique layer name' issue
      layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
      
    model.input._name = 'ensemble_in' + str(i+1) 
    ensemble_visible.append(model.input)

  ensemble_outputs = [model.output for model in members]
  output = layers.Average()(ensemble_outputs)
  model = models.Model(inputs=ensemble_visible, outputs=output)
  
  #because this model will only be used to make predictions, it does't matter wich optimizer I use
  model.compile(loss='sparse_categorical_crossentropy',) 
  return model

stacked_model = defineStackedModel(members) #this has 20 inputs, one output

stacked_model.save('/modelFinal.h5')
