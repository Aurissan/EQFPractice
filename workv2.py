###this code is a mess

import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as dt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential




def build_model(layers): #keras LSTM network
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    #start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    #print("> Compilation Time : ", time.time() - start)
    return model

# def predict_point_by_point(model, data):
#     #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
#     predicted = model.predict(data)
#     predicted = np.reshape(predicted, (predicted.size,))
#     return predicted
#
# def predict_sequence_full(model, data, window_size):
#     #Shift the window by 1 new prediction each time, re-run predictions on new window
#     curr_frame = data[0]
#     predicted = []
#     for i in range(len(data)):
#         predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
#         curr_frame = curr_frame[1:]
#         curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
#     return predicted
#
# def predict_sequences_multiple(model, data, window_size, prediction_len):
#     #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
#     prediction_seqs = []
#     for i in range(int(len(data)/prediction_len)):
#         curr_frame = data[i*prediction_len]
#         predicted = []
#         for j in range(prediction_len):
#             predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
#             curr_frame = curr_frame[1:]
#             curr_frame = np.insert(curr_frame, window_size-1, predicted[-1], axis=0)
#         prediction_seqs.append(predicted)
#     return prediction_seqs
#
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

back_fat=pd.read_excel('tables.xls', sheet_name=3, header=None)  #preprocess input table
back_fat.dropna(inplace=True)
back_fat.columns=['x','y']
b2=back_fat.loc[(back_fat['x']>='2014-10-31')].copy(deep=True) #split data before and after the 2014 crisis
b1=back_fat.loc[(back_fat['x']<'2014-10-31')].copy(deep=True)
#back_fat=back_fat.loc[(back_fat['x']>='2014-10-31')].copy(deep=True)
scaler=preprocessing.MinMaxScaler() #normalize data
t1=b1['y'].values
t1=t1.reshape(-1,1)
t1=scaler.fit_transform(t1)
t1.reshape(-1)
b1['y']=t1
t2=b2['y'].values
t2=t2.reshape(-1,1)
t2=scaler.fit_transform(t2)
t2.reshape(-1)
b2['y']=t2
test=[b1,b2]
back_fat=pd.concat(test)

# EMA=0.0
# gamma=1
# smoothe=back_fat['y'].values
# smoothe=smoothe.astype(np.float64)
# for x in np.nditer(smoothe, op_flags=['readwrite']):
#     EMA=gamma*x+(1-gamma)*EMA
#     x[...]=EMA
#
# back_fat['y']=smoothe

#back_fat['date']=back_fat['date'].apply(lambda x: x.toordinal())
#new_back=back_fat.loc[(back_fat['x']>='2014-10-31')&(back_fat['x']<='2018-01-13')].copy(deep=True)

new_back=back_fat.loc[(back_fat['x']<='2018-08-21')].copy(deep=True)  #split into testing and training datasets
back_predict=back_fat.loc[back_fat['x']>='2018-08-28'].copy(deep=True)
new_back['x']=new_back['x'].apply(lambda x: x.toordinal()) #convert date
back_predict['x']=back_predict['x'].apply(lambda x: x.toordinal())



backarr=back_predict['x'].values
backarr=backarr.astype(np.float64)
for elem in np.nditer(backarr, op_flags=['readwrite']):
    elem[...]=elem-733000 #just to make the graphs a bit more readable
    #elem[...] = elem / 10000
backarr=backarr.reshape(-1,1)
#backarr=np.divide(backarr, 10000)

backarr2=new_back['x'].values
backarr2=backarr2.astype(np.float64)
for elem in np.nditer(backarr2, op_flags=['readwrite']):
    elem[...]=elem-733000
    #elem[...]=elem/10000
backarr2=backarr2.reshape(-1,1)
#backarr2=np.divide(backarr2, 10000)

outarr=new_back['y'].values
outarr=outarr.astype(np.float64)
outarr=outarr.reshape(-1,1)
outarr2=back_predict['y'].values
outarr2=outarr2.astype(np.float64)
outarr2=outarr2.reshape(-1,1)
# plt.plot(back_fat['x'], back_fat['y'])
# plt.plot(back_fat['x'], smoothe)
# plt.show()
#print(backarr.shape[0])
#train=create_dataset(backarr2)
#train1,train2=create_dataset(outarr)
#test1,test2=create_dataset(outarr2)
#train1 = np.reshape(train1, (train1.shape[0], 1, train1.shape[1]))
#test1=np.reshape(test1, (test1.shape[0], 1, test1.shape[1]))

#backarr2=np.reshape(backarr2, (backarr2.shape[0], 1, backarr2.shape[1]))
#backarr=np.reshape(backarr, (backarr.shape[0], 1, backarr.shape[1]))

#model= build_model([1,1,100,1])
#model.fit(backarr2, outarr, batch_size=1, epochs=50, validation_split=0.2, verbose=0)
#predicted = predict_sequence_full(model, test, np.size(backarr))
#predict=predict_sequences_multiple(model, backarr, 2, backarr.shape[0])

# backarr=backarr.reshape(-1,1)
# plt.plot(backarr,outarr2)
# plt.plot(backarr,predict)
# plt.show()
# backarr2=backarr2.reshape(-1,1)

#backarr=backarr.reshape(-1,1)
#print(predict.shape," " ,outarr2.shape, " ", train1.shape, train2.shape, backarr2.shape)
#classificator=MLPRegressor(hidden_layer_sizes=(200, ), activation='relu', solver='lbfgs', alpha=0.0001,
#batch_size='auto', learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=200,
#shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


regressor=KernelRidge(alpha=0.005, kernel='rbf', gamma=0.00025, degree=3, coef0=1, kernel_params=None) #sklearn NN models
regressor2=SVR(C=1e3,epsilon=0.01,gamma=0.001,tol=1e-4)
#classificator.fit(backarr2,outarr)
#print(classificator.score(backarr, back_predict['y']))
#print(classificator.score(backarr2, new_back['y']))


# out=[]
# scores=[]
# for x,y in zip(backarr, outarr2):
#     x=x.reshape(1, -1)
#     backarr2=backarr2.reshape(-1, 1)
#     regressor2.fit(backarr2, outarr)
#     mid=regressor2.predict(x)
#     #mid2=classificator.score(x, y)
#     #print (x, y)
#     backarr2=np.append(backarr2, x)
#     outarr=np.append(outarr, y)
#     out=np.append(out, mid)
#     #scores=np.append(scores,mid2)


regressor.fit(backarr2,outarr)
out=regressor.predict(backarr2)
regressor2.fit(backarr2,outarr)
out2=regressor2.predict(backarr2)
res=regressor.score(backarr2,outarr)
res2=regressor2.score(backarr2,outarr)

time1,time2=[],[]
for i in range(dt.date(2008,2,25).toordinal(),dt.date(2014,2,3).toordinal(),7): #make an array of mondays
    time1=np.append(time1,i)
for i in range(dt.date(2014,10,27).toordinal(),dt.date(2018,8,21).toordinal(),7): time2=np.append(time2,i)
for i in np.nditer(time1, op_flags=['readwrite']): i[...]=i-733000
for i in np.nditer(time2, op_flags=['readwrite']): i[...]=i-733000
time1=time1.reshape(-1,1)
time2=time2.reshape(-1,1)
interpol=regressor.predict(time1) #interpolate the original data that has gaps in it to fill the said gaps
interpol2=regressor.predict(time2)
time=np.concatenate((time1,time2))
result=np.concatenate((interpol,interpol2))

# plt.plot(time,result)
# plt.plot(backarr2, outarr)
# plt.show()

trainx,trainy=create_dataset(result) #probably for keras LSTM
trainx=trainx.reshape(-1,1)
trainy=trainy.reshape(-1,1)


pred=[]
for i in range(dt.date(2018,8,27).toordinal(),dt.date(2018,10,8).toordinal(),7):
    pred=np.append(pred,i)
for i in np.nditer(pred, op_flags=['readwrite']): i[...]=i-733000


#predictions of NN models
pred=pred.reshape(-1,1)
tes=regressor.predict(pred)
# plt.plot(backarr, outarr2)
# plt.plot(pred,tes)
# plt.show()
#print(pred.shape)


first=[]
first=np.append(first,trainy[-1])
# first=np.expand_dims(first[0], axis=0)
# first=first.reshape(-1,1)
# first = np.reshape(first, (first.shape[0], 1,first.shape[1]))
trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))

look_back=1
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx,trainy,  epochs=1, batch_size=1, verbose=2)

#b=model.predict(first)
weeks=5
for i in range(0,weeks):
    x=[]
    x=np.append(x,first[i])
    x=np.expand_dims(x[0],axis=0)
    x=x.reshape(-1,1)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    ouch=model.predict(x)
    first=np.append(first,ouch)


deito=[] #convert back to date
for x in np.nditer(pred, op_flags=['readwrite']):
    x[...]+=733000
    r = dt.date.fromordinal(x)
    deito=np.append(deito,r)
#print(deito)
first=first.reshape(-1,1)
test=tes.reshape(-1,1)
tes=scaler.inverse_transform(tes)
first=scaler.inverse_transform(first)
#plt.plot(backarr,outarr2)

predict=[]
for x in np.nditer(backarr, op_flags=['readwrite']):
    x[...] += 733000
    r = dt.date.fromordinal(x)
    predict=np.append(predict,r)
outarr2=scaler.inverse_transform(outarr2)

#plot
plt.plot(deito,first)
plt.plot(deito, tes)
plt.plot(predict,outarr2)

plt.show()

##as expected, nothing good came out of this ""project""