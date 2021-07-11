# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:44:32 2021

@author: Febrian
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

Datasets=pd.read_csv('Datasets2.csv',sep=';')
Airfoil=pd.read_csv('Airfoil Datasets.csv',sep=';')
df=Airfoil.merge(Datasets, on='Airfoil')

X = df.iloc[:, 1:104]
y = df.iloc[:, 104]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

Xselig=X[5209:, :]
yselig=y[5209:]

X=X[:5208, :]
y=y[:5208]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


def build_model(): 
    model = Sequential()
    model.add(Dense(16, activation='tanh', input_shape=[X_train.shape[1]]))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(100, activation=tf.keras.layers.LeakyReLU()))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    optimizers=tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse', optimizer=optimizers, metrics=['mae', 'mse'])
    return model
    
model=build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.',end='')
        
EPOCHS=1000

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',patience=200)    

history = model.fit(X_train, y_train, epochs=EPOCHS, 
                    validation_split=0.2, verbose=0, 
                    callbacks=[early_stop, PrintDot()])

def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    
    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'],hist['mae'],
             label = 'Training Error')
    plt.plot(hist['epoch'],hist['val_mae'],
             label = 'Validation Error')
    plt.title('Mean Absolute Error = {:5.6f}'.format(hist['mae'][len(hist)-1]))
    plt.legend()
    
    plt.figure(2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'],hist['mse'],
             label = 'Training Error')
    plt.plot(hist['epoch'],hist['val_mse'],
             label = 'Validation Error')
    plt.title('Mean Square Error = {:5.6f}'.format(hist['mse'][len(hist)-1]))
    plt.legend()

plot_history(history)

loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)

print("\nTesting Mean Abs Error: {:5.6f} ".format(mae))

test_predictions = model.predict(X_test).flatten()
plt.figure(3)
plt.scatter(y_test, test_predictions,s=12,c='orange')
plt.xlabel('True Value')
plt.ylabel('Prediction')
maePred=(abs(test_predictions-y_test)).sum()/len(test_predictions)
msePred=((test_predictions-y_test)**2).sum()/len(test_predictions)
plt.title("MAE = {:5.6f}, MSE = {:5.6f} ".format(maePred,msePred))
plt.axis('equal')
plt.axis('square')
plt.plot([-100,100],[-100,100])


selig_prediction=model.predict(Xselig).flatten()
plt.figure(4)
plt.scatter(yselig, selig_prediction,s=12,c='orange')
plt.xlabel('True Value')
plt.ylabel('Prediction')
maeSelig=(abs(selig_prediction-yselig)).sum()/len(selig_prediction)
mseSelig=((selig_prediction-yselig)**2).sum()/len(selig_prediction)
plt.title("MAE = {:5.6f}, MSE = {:5.6f} ".format(maeSelig,mseSelig))
plt.axis('equal')
plt.axis('square')
plt.plot([-100,100],[-100,100])

