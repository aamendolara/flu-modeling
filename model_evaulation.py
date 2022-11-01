import pandas as pd
from functions import *
from model_functions import *
from the_model import *
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#F4511E", "#FB8C00", "#FFB300", "#FDD835", "#C0CA33", "#7CB342", "#43A047", "#00897B", "#00ACC1","#3949AB"])


#########################################################################################################################
#prep data for 4wk time lag recursive prediction full dataset no future temp
series_data = data_load('C:/Users/Alfred/Desktop/code/VariousThesisStuff_Alfred/more_combined_data.csv',4, futuret = False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

#build models 
#540 week split with full data
#predicts data3 timelag
model_1_1TF = threeway_split_model(data1,labels1,data2,labels2,data3,labels30, prediction_type=1)
#predicts data2 
model_4_1TF = threeway_split_model(data1,labels1,data3,labels3,data2,labels20, prediction_type=1)
#predicts data1 
model_7_1TF = threeway_split_model(data2,labels2,data3,labels3,data1,labels10, prediction_type=1)

plt.plot(model_1_1TF[0])
plt.plot(labels3, color='black' )
#########################################################################################################################
#prep data for 4wk time lag recursive prediction reduced dataset no future temp
series_data = data_load(4,futuret = False,  full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

#build models 
#540 week split with full data
#predicts data3 timelag
model_1_1FF = threeway_split_model(data1,labels1,data2,labels2,data3,labels30, prediction_type=1, full= False)
#predicts data2 
model_4_1FF = threeway_split_model(data1,labels1,data3,labels3,data2,labels20, prediction_type=1, full= False)
#predicts data1 
model_7_1FF = threeway_split_model(data2,labels2,data3,labels3,data1,labels10, prediction_type=1, full= False)


#yes = model_1_1FF[0]
plt.plot(model_1_1FF[0])
plt.plot(labels3, color='black')

#########################################################################################################################
#prep data recursive prediction without time lag--this data contains full dataset, no future temp
series_data = data_load(1, futuret =False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

#540 weeks
#data3
#no time lag
model_2_1TF = threeway_split_model(data1,labels1,data2,labels2,data3,labels30, prediction_type=2)
#data2
model_5_1TF = threeway_split_model(data1,labels1,data3,labels3,data2,labels20, prediction_type=2)
#predicts data1 
model_8_1TF = threeway_split_model(data2,labels2,data3,labels3,data1,labels10, prediction_type=2)

week20_model = threeway_split_model(data1, labels1, data2,labels2,data3,labels30, prediction_type=5)



#plt.plot(model_5_1TF[0])
#plt.plot(labels2, color='black')

#predi = model_5_1TF[0]

#plt.scatter(labels20[:,0],predi[:,0])
#plt.scatter(labels20[:,4],predi[:,4])
#plt.scatter(labels20[:,9],predi[:,9], color="#C0CA33")


#########################################################################################################################
#base model - future temp data & full data set
#no time lag
#predicts out to 10 weeks, no recursive function 
series_data = data_load(1, full = False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

#400 split middle
data_1_2 = pd.DataFrame(data=series_data.iloc[0:200])
data_2_2 = pd.DataFrame(data=series_data.iloc[200:600])
data_3_2 = pd.DataFrame(data=series_data.iloc[600:806])

[data1a,data2a,data3a,labels1a,labels2a,labels3a,labels10a,labels20a,labels30a] = data_maker(data_1_2, data_2_2, data_3_2)

#400 split middle
data_1_3 = pd.DataFrame(data=series_data.iloc[0:300])
data_2_3 = pd.DataFrame(data=series_data.iloc[300:700])
data_3_3 = pd.DataFrame(data=series_data.iloc[700:806])

[data1b,data2b,data3b,labels1b,labels2b,labels3b,labels10b,labels20b,labels30b] = data_maker(data_1_3, data_2_3, data_3_3)

#400 split middle
data_1_4 = pd.DataFrame(data=series_data.iloc[0:100])
data_2_4 = pd.DataFrame(data=series_data.iloc[100:500])
data_3_4 = pd.DataFrame(data=series_data.iloc[500:806])

[data1c,data2c,data3c,labels1c,labels2c,labels3c,labels10c,labels20c,labels30c] = data_maker(data_1_4, data_2_4, data_3_4)

#700 split
data_1_5 = pd.DataFrame(data=series_data.iloc[0:700])
data_2_5 = pd.DataFrame(data=series_data.iloc[700:806])

[data1d,data2d,labels1d,labels2d,labels10d,labels20d] = data_maker2(data_1_5, data_2_5)

#700 split
data_1_6 = pd.DataFrame(data=series_data.iloc[0:106])
data_2_6 = pd.DataFrame(data=series_data.iloc[106:806])

[data1e,data2e,labels1e,labels2e,labels10e,labels20e] = data_maker2(data_1_6, data_2_6)

#700 split 
data_1_7 = pd.DataFrame(data=series_data.iloc[0:53])
data_2_7 = pd.DataFrame(data=series_data.iloc[53:753])
data_3_7 = pd.DataFrame(data=series_data.iloc[753:806])

[data1f,data2f,data3f,labels1f,labels2f,labels3f,labels10f,labels20f,labels30f] = data_maker(data_1_7, data_2_7, data_3_7)


#540 weeks
model_3_1 = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_6_1 = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_9_1 = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)

#400 week
model_3_2 = twoway_prediction_model(data2a, labels20a, data1a, labels10a, data3a, labels30a, prediction_type=3, full= False)
model_6_2 = twoway_prediction_model(data2b, labels20b, data1b, labels10b, data3b, labels30b, prediction_type=3, full= False)
model_9_2 = twoway_prediction_model(data2c, labels20c, data1c, labels10c, data3c, labels30c, prediction_type=3, full= False)

#700 week 
model_7_3 = twoway_data(data1d, labels10d, data2d, labels20d, prediction_type=3, full= False)
model_8_3 = twoway_data(data2e, labels20e, data1e, labels10e, prediction_type=3, full= False)
model_9_3 = twoway_prediction_model(data2f, labels20f, data1f, labels10f, data3f, labels30f, prediction_type=3, full= False)


#plt.plot(model_3_1[0])
#plt.plot(labels3, color='black')
#########################################################################################################################
#base model - future temp data & reduced data set
#no time lag
#predicts out to 10 weeks, no recursive function 
series_data = data_load(1, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)


#540 weeks
model_3_1TF = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_6_1TF = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_9_1TF = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)


#########################################################################################################################
#base model - future temp data & reduced data set
#no time lag
#predicts out to 10 weeks, no recursive function 
series_data = data_load(1, futuret = False, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)


#540 weeks
model_3_1FF = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_6_1FF = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_9_1FF = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)

base_model_predictions = threeway_split_model(data1,labels1,data2,labels2,data3,labels3, prediction_type=4, full= False)
base_model_predictions2 = threeway_split_model(data1,labels1,data3,labels3,data2,labels2, prediction_type=4, full= False)
base_model_predictions3 = threeway_split_model(data2,labels2,data3,labels3,data1,labels1, prediction_type=4, full= False)

########################################################################################################################################

## standard model with 12 week, 16 week, and 52 week time lag - reduced dataset, future temp
series_data = data_load(12, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)


model_12lag = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_12lag2 = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_12lag3 = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)

series_data = data_load(16, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)


model_16lag = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_16lag2 = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_16lag3 = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)

series_data = data_load(52, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

model_52lag = threeway_split_model(data1,labels10,data2,labels20,data3,labels30, prediction_type=3, full= False)
model_52lag2 = threeway_split_model(data1,labels10,data3,labels30,data2,labels20, prediction_type=3, full= False)
model_52lag3 = threeway_split_model(data2,labels20,data3,labels30,data1,labels10, prediction_type=3, full= False)
########################################################################################################################################

#4 week lag without future temperature -- show importance of future climate data
series_data = data_load(4, futuret = False, full=False)

#break up data sets and create labels
#540 split
data_1_1 = pd.DataFrame(data=series_data.iloc[0:270])
data_2_1 = pd.DataFrame(data=series_data.iloc[270:540])
data_3_1 = pd.DataFrame(data=series_data.iloc[540:806])

[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)


model_4lag_noT = threeway_split_model(data1,labels1,data2,labels2,data3,labels30, prediction_type=1, full=False)


#####################################################################################################################################
moddata = data_load(4, futuret=False, full=False,outliers=False)

#regular vs no outlier 
data_train = pd.DataFrame(data=moddata.iloc[0:500])
data_test = pd.DataFrame(data=moddata.iloc[500:])
data_train2 = pd.DataFrame(data=moddata.iloc[150:])

[data_t,data_t2,labels_t,labels_t2,labels_t10,labels_t20] = data_maker2(data_train, data_test)

no_outliers = twoway_data(data_t,labels_t,data_t2,labels_t20, prediction_type = 1, full= False)

#nope = no_outliers[0]
#plt.plot(no_outliers[0])
#plt.plot(labels_t2, color='black')
