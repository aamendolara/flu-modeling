
#set seed to reduce randomness
from numpy.random import seed
seed(1)
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#F4511E", "#FB8C00", "#FFB300", "#FDD835", "#C0CA33", "#7CB342", "#43A047", "#00897B", "#00ACC1","#3949AB"])

moddata = data_load(1, futuret=False, full=False,outliers=False)
fulldata = data_load(1, futuret=False, full=False)
moddata.shape

#regular dataset 
data_1_1 = pd.DataFrame(data=fulldata.iloc[0:270])
data_2_1 = pd.DataFrame(data=fulldata.iloc[270:540])
data_3_1 = pd.DataFrame(data=fulldata.iloc[540:806])



[data1,data2,data3,labels1,labels2,labels3,labels10,labels20,labels30] = data_maker(data_1_1, data_2_1, data_3_1)

#outliers removed
data_no_outlier1 = pd.DataFrame(data=moddata.iloc[0:216])
data_no_outlier2 = pd.DataFrame(data=moddata.iloc[216:432])
data_no_outlier3 = pd.DataFrame(data=moddata.iloc[432:648])

[data1no,data2no,data3no,labels1no,labels2no,labels3no,labels10no,labels20no,labels30no] = data_maker(data_no_outlier1, data_no_outlier2, data_no_outlier3)


#######################################
#train 2 models 
#standard model with full data 
tf.keras.backend.clear_session()


standard_model = build_modelE(data1, labels1)
standard_model.fit(data1,
                   labels1,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

standard_model.fit(data2,
                   labels2,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

predictions = predict_future_r10(standard_model, data3)

#no outlier model

outlier_model = build_modelE(data1no, labels1no)
outlier_model.fit(data1no,
                  labels1no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_model.fit(data2no,
                  labels2no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_predictions = predict_future_r10(outlier_model, data3no)


outlier_predictions2 = predict_future_r10(outlier_model, data3)

plot_test0 = plt.figure()
plt.subplot(211)
plt.plot(outlier_predictions2)
plt.plot(labels3, color='black')
plt.title('Outliers removed')
plt.subplot(212)


plt.plot(predictions)
plt.plot(labels3, color='black')
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/set310wksFINAL.svg')
plt.title('Standard Model')


signal_signs = outlier_predictions2 - predictions
signal = abs(outlier_predictions2 -predictions)

signal_df = pd.DataFrame({'wk1':signal[:,0], 'wk2':signal[:,1], 'wk3':signal[:,2], 
                          'wk4':signal[:,3], 'wk5':signal[:,4], 'wk6':signal[:,5], 
                          'wk7':signal[:,6], 'wk8':signal[:,7], 'wk9':signal[:,8],
                          'wk10':signal[:,9]})

stats = signal_df.apply(pd.DataFrame.describe, axis=1)


##################################################

standard_model1 = build_modelE(data1, labels1)
standard_model1.fit(data1,
                   labels1,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

standard_model1.fit(data3,
                   labels3,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

predictions1 = predict_future_r10(standard_model, data2)

#no outlier model

outlier_model1 = build_modelE(data1no, labels1no)
outlier_model1.fit(data1no,
                  labels1no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_model1.fit(data3no,
                  labels3no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_predictions1 = predict_future_r10(outlier_model, data2no)


outlier_predictions21 = predict_future_r10(outlier_model, data2)

plot_test1 = plt.figure()
plt.subplot(211)
plt.plot(outlier_predictions21)
plt.plot(labels3, color='black')
plt.title('Outliers removed')
plt.subplot(212)
plt.plot(predictions1)
plt.plot(labels3, color='black')
plt.title('Standard Model')




signal_signs1 = outlier_predictions21 - predictions1
signal1 = abs(outlier_predictions21 - predictions1)

signal_df1 = pd.DataFrame({'wk1':signal1[:,0], 'wk2':signal1[:,1], 'wk3':signal1[:,2], 
                           'wk4':signal1[:,3], 'wk5':signal1[:,4], 'wk6':signal1[:,5], 
                           'wk7':signal1[:,6], 'wk8':signal1[:,7], 'wk9':signal1[:,8],
                           'wk10':signal1[:,9]})

stats1 = signal_df1.apply(pd.DataFrame.describe, axis=1)


#####################################################
standard_model2 = build_modelE(data2, labels2)
standard_model2.fit(data2,
                   labels2,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

standard_model2.fit(data3,
                   labels3,
                   epochs=500,
                   batch_size=270,
                   validation_split=0.1,
                   verbose=1,
                   shuffle=False,
                   callbacks=[reduce_lr, early_stop])

predictions0 = predict_future_r10(standard_model, data1)

#no outlier model

outlier_model2 = build_modelE(data2no, labels2no)
outlier_model2.fit(data2no,
                  labels2no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_model2.fit(data3no,
                  labels3no,
                  epochs=500,
                  batch_size=270,
                  validation_split=0.1,
                  verbose=1,
                  shuffle=False,
                  callbacks=[reduce_lr, early_stop])

outlier_predictions2 = predict_future_r10(outlier_model, data1no)


outlier_predictions22 = predict_future_r10(outlier_model, data1)

plot_test2 = plt.figure()
plt.subplot(211)
plt.plot(outlier_predictions22)
plt.plot(labels1, color='black')
plt.title('Outliers removed')
plt.subplot(212)
plt.plot(predictions0)
plt.plot(labels1, color='black')
plt.title('Standard Model')


signal_signs2 = outlier_predictions22 - predictions0
signal2 = abs(outlier_predictions22 - predictions0)

signal_df2 = pd.DataFrame({'wk1':signal2[:,0], 'wk2':signal2[:,1], 'wk3':signal2[:,2], 
                           'wk4':signal2[:,3], 'wk5':signal2[:,4], 'wk6':signal2[:,5], 
                           'wk7':signal2[:,6], 'wk8':signal2[:,7], 'wk9':signal2[:,8],
                           'wk10':signal2[:,9]})

stats2 = signal_df2.apply(pd.DataFrame.describe, axis=1)


##########################################
year =[13,65,117,169,221]
year = [x - 4 for x in year]
year1 =[273,325,377,429,481,533] 
year1 = [x - 270 for x in year1]  
year2 =[585,637,689,741,793]
year2 = [x - 544 for x in year2]


plt.plot(outlier_predictions2)
plt.plot(signal_signs)
plt.plot(labels3, color='black')
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/signal.svg')


plt.plot(signal1)
plt.plot(labels2, color='black')


plt.plot(labels1, color='black')
plt.plot(signal2[:,9])
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/signal.svg')


plt.scatter(labels10[:,0], predictions0[:,0])
plt.scatter(labels10[:,4], predictions0[:,4])
plt.scatter(labels10[:,9], predictions0[:,9], color="#C0CA33")
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/series_errorscatter.svg')

plt.scatter(labels20[:,0], predictions1[:,0])
plt.scatter(labels20[:,4], predictions1[:,4])
plt.scatter(labels20[:,9], predictions1[:,9], color="#C0CA33")
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/series2_errorscatter.svg')

plt.scatter(labels30[:,0], predictions[:,0])
plt.scatter(labels30[:,4], predictions[:,4])
plt.scatter(labels30[:,9], predictions[:,9], color="#C0CA33")
plt.savefig('C:/Users/eric_fortune/Desktop/THESIS Backup/Charts_tables_graphs/Predictions/series3_errorscatter.svg')


