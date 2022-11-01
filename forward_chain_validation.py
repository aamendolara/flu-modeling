# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:01:40 2022

@author: Alfred
"""

import pandas as pd
from functions import *
from model_functions import *
from the_model import *
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#F4511E", "#FB8C00", "#FFB300", "#FDD835", "#C0CA33", "#7CB342", "#43A047", "#00897B", "#00ACC1","#3949AB"])


#this is new code to properly validate this model using forward chaining or last block evaluation 
#########################################################################################################################
#prep data for 4wk time lag recursive prediction full dataset no future temp
series_data = data_load('C:/Users/Alfred/Desktop/code/VariousThesisStuff_Alfred/more_combined_data.csv',4, futuret = False, full = False)

#break up data sets and create labels this will be broken into 10 segments and predict based on 1 -> predict 2, 1 + 2 -> predict 3.....etc 
#1 predicting 2
data_1_1 = pd.DataFrame(data=series_data.iloc[0:80])
data_1_2 = pd.DataFrame(data=series_data.iloc[80:160])

[data1,data2,labels1,labels2,labels10,labels20] = data_maker2(data_1_1, data_1_2)

model_1 = basic_prediction_model(data1, labels1, data2, labels10, prediction_type=1, full= False ) 

model_1[1]
plt.plot(model_1[0])
plt.plot(labels2, color='black' )

#1, 2 predicting 3
data_2_1 = pd.DataFrame(data=series_data.iloc[0:160])
data_2_2 = pd.DataFrame(data=series_data.iloc[160:240])

[data12,data22,labels12,labels22,labels102,labels202] = data_maker2(data_2_1, data_2_2)

model_2 = basic_prediction_model(data12, labels12, data22, labels102, prediction_type=1, full= False ) 

model_2[1]
plt.plot(model_2[0])
plt.plot(labels22, color='black' )


#1, 2, 3 predicting 4
data_3_1 = pd.DataFrame(data=series_data.iloc[0:240])
data_3_2 = pd.DataFrame(data=series_data.iloc[240:320])

[data13,data23,labels13,labels23,labels103,labels203] = data_maker2(data_3_1, data_3_2)

model_3 = basic_prediction_model(data13, labels13, data23, labels103, prediction_type=1, full= False ) 

model_3[1]
plt.plot(model_3[0])
plt.plot(labels23, color='black' )

#1, 2, 3, 4 predicting 5
data_4_1 = pd.DataFrame(data=series_data.iloc[0:320])
data_4_2 = pd.DataFrame(data=series_data.iloc[320:400])

[data14,data24,labels14,labels24,labels104,labels204] = data_maker2(data_4_1, data_4_2)

model_4 = basic_prediction_model(data14, labels14, data24, labels104, prediction_type=1, full= False ) 

model_4[1]
plt.plot(model_4[0])
plt.plot(labels24, color='black' )


#1, 2, 3, 4, 5 predicting 6
data_5_1 = pd.DataFrame(data=series_data.iloc[0:400])
data_5_2 = pd.DataFrame(data=series_data.iloc[400:480])

[data15,data25,labels15,labels25,labels105,labels205] = data_maker2(data_5_1, data_5_2)

model_5 = basic_prediction_model(data15, labels15, data25, labels105, prediction_type=1, full= False ) 

model_5[1]
plt.plot(model_5[0])
plt.plot(labels25, color='black' )


#1, 2, 3, 4, 5, 6 predicting 7
data_6_1 = pd.DataFrame(data=series_data.iloc[0:480])
data_6_2 = pd.DataFrame(data=series_data.iloc[480:560])

[data16,data26,labels16,labels26,labels106,labels206] = data_maker2(data_6_1, data_6_2)

model_6 = basic_prediction_model(data16, labels16, data26, labels106, prediction_type=1, full= False ) 

model_6[1]
plt.plot(model_6[0])
plt.plot(labels26, color='black' )


#1, 2, 3, 4, 5, 6, 7 predicting 8
data_7_1 = pd.DataFrame(data=series_data.iloc[0:560])
data_7_2 = pd.DataFrame(data=series_data.iloc[560:640])

[data17,data27,labels17,labels27,labels107,labels207] = data_maker2(data_7_1, data_7_2)

model_7 = basic_prediction_model(data17, labels17, data27, labels107, prediction_type=1, full= False ) 

model_7[1]
plt.plot(model_7[0])
plt.plot(labels27, color='black' )

#1, 2, 3, 4, 5, 6, 7, 8 predicting 9
data_8_1 = pd.DataFrame(data=series_data.iloc[0:640])
data_8_2 = pd.DataFrame(data=series_data.iloc[640:720])

[data18,data28,labels18,labels28,labels108,labels208] = data_maker2(data_8_1, data_8_2)

model_8 = basic_prediction_model(data18, labels18, data28, labels108, prediction_type=1, full= False ) 

model_8[1]
plt.plot(model_8[0])
plt.plot(labels28, color='black' )

#1, 2, 3, 4, 5, 6, 7, 8, 9 predicting 10
data_9_1 = pd.DataFrame(data=series_data.iloc[0:720])
data_9_2 = pd.DataFrame(data=series_data.iloc[720:800])

[data19,data29,labels19,labels29,labels109,labels209] = data_maker2(data_9_1, data_9_2)

model_9 = basic_prediction_model(data19, labels19, data29, labels29, prediction_type=4, full= False ) 

model_9[1]
plt.plot(model_9[0])
plt.plot(labels29, color='black' )


