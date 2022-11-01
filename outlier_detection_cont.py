import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from functions import *
###############
#signal2 --> data1 predictions / lo
#signal --> data2 predictions / lo2
#signal1 --> data3 predictions / lo3
###############


#encode outlier year
lo = np.zeros((270,3))
lo[0:22,0] = 1
lo[22:210,1] = 1
lo[210:240,2] = 1
lo[240:270,1] = 1
 
lo2 = np.zeros((270,3))
lo2[38:60,0] = 1
lo2[18:30,0] = 1
lo2[208:230,2] = 1
lo2[60:208,1] = 1
lo2[230:270,1] = 1

lo3 = np.zeros((266,3))
lo3[190:226,0] = 1
lo3[0:38,1] = 1
lo3[65:190,1] = 1
lo3[226:245,1] = 1
lo3[245:266,2] = 1
lo3[38:65,2] = 1


#simple binar classifcation model
def build_model_outlier(input_shape):
    model = keras.Sequential([
            layers.Dense(4, activation='relu', input_shape=(input_shape)),
            layers.Dense(4, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
    
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return model 

tf.keras.backend.clear_session()

outlier_predictor = build_model_outlier((10,))


outlier_predictor.fit(signal2,
                  lo,
                  epochs=1000,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_predictor.fit(signal,
                  lo2,
                  epochs=1000,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_predictor.fit(signal1,
                  lo3,
                  epochs=1000,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])


#test predictions

predictions_alpha = outlier_predictor.predict(signal1)
predictions_beta = outlier_predictor.predict(signal2)
predictions_gamma = outlier_predictor.predict(signal)

predict_data3 = plt.figure()
plt.plot(predictions_alpha[:,1])
plt.plot(lo3[:,1])
plt.plot(labels3)

predict_data1 = plt.figure()
plt.plot(predictions_beta[:,0])
plt.plot(lo[:,0])
plt.plot(labels1)

predict_data2 = plt.figure()
plt.plot(predictions_gamma[:,0])
plt.plot(lo2[:,0])
plt.plot(labels2)




