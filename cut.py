##basically the same code as the workv2.py but for different data and cleaner, so for the comments look up workv2




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

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


cut_fat=pd.read_excel('tables.xls', sheet_name=4, header=None)
cut_fat.dropna(inplace=True)
cut_fat.columns=['x','y']
c2=cut_fat.loc[(cut_fat['x']>='2014-10-31')].copy(deep=True)
c1=cut_fat.loc[(cut_fat['x']<'2014-10-31')].copy(deep=True)
scaler=preprocessing.MinMaxScaler()
t1=c1['y'].values
t1=t1.reshape(-1,1)
t1=scaler.fit_transform(t1)
t1.reshape(-1)
c1['y']=t1
t2=c2['y'].values
t2=t2.reshape(-1,1)
t2=scaler.fit_transform(t2)
t2.reshape(-1)
c2['y']=t2
test=[c1,c2]
cut_fat=pd.concat(test)
new_cut=cut_fat.loc[(cut_fat['x']<='2018-08-21')].copy(deep=True)
cut_predict=cut_fat.loc[cut_fat['x']>='2018-08-28'].copy(deep=True)
new_cut['x']=new_cut['x'].apply(lambda x: x.toordinal())
cut_predict['x']=cut_predict['x'].apply(lambda x: x.toordinal())


cutarr=new_cut['x'].values
cutarr=cutarr.astype(np.float64)
for x in np.nditer(cutarr,op_flags=['readwrite']):
    x[...]-=733000
cutarr=cutarr.reshape(-1,1)

cutarr2=cut_predict['x'].values
cutarr2=cutarr2.astype(np.float64)
for x in np.nditer(cutarr2,op_flags=['readwrite']):
    x[...]-=733000
cutarr2=cutarr2.reshape(-1,1)

outarr=new_cut['y'].values
outarr=outarr.astype(np.float64)
outarr=outarr.reshape(-1,1)
outarr2=cut_predict['y'].values
outarr2=outarr2.astype(np.float64)
outarr2=outarr2.reshape(-1,1)

regressor=KernelRidge(alpha=0.005, kernel='rbf', gamma=0.00025, degree=3, coef0=1, kernel_params=None)
regressor2=SVR(C=1e3,epsilon=0.01,gamma=0.001,tol=1e-4)

regressor.fit(cutarr,outarr)

time1,time2=[],[]
for i in range(dt.date(2008,2,25).toordinal(),dt.date(2014,2,3).toordinal(),7):
    time1=np.append(time1,i)
for i in range(dt.date(2014,10,27).toordinal(),dt.date(2018,8,21).toordinal(),7): time2=np.append(time2,i)
for i in np.nditer(time1, op_flags=['readwrite']): i[...]=i-733000
for i in np.nditer(time2, op_flags=['readwrite']): i[...]=i-733000
time1=time1.reshape(-1,1)
time2=time2.reshape(-1,1)
interpol=regressor.predict(time1)
interpol2=regressor.predict(time2)
time=np.concatenate((time1,time2))
result=np.concatenate((interpol,interpol2))
# plt.plot(cutarr,outarr)
# plt.plot(time,result)
# plt.show()
trainx,trainy=create_dataset(result)
trainx=trainx.reshape(-1,1)
trainy=trainy.reshape(-1,1)

pred=[]
for i in range(dt.date(2018,8,27).toordinal(),dt.date(2018,10,8).toordinal(),7):
    pred=np.append(pred,i)
for i in np.nditer(pred, op_flags=['readwrite']): i[...]=i-733000

pred=pred.reshape(-1,1)
tes=regressor.predict(pred)

first=[]
first=np.append(first,trainy[-1])
# first=np.expand_dims(first[0], axis=0)
# first=first.reshape(-1,1)
# first = np.reshape(first, (first.shape[0], 1,first.shape[1]))
trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))

look_back=1
model = Sequential()
model.add(LSTM(10, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx,trainy,  epochs=2, batch_size=1)

weeks=5
for i in range(0,weeks):
    x=[]
    x=np.append(x,first[i])
    x=np.expand_dims(x[0],axis=0)
    x=x.reshape(-1,1)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    ouch=model.predict(x)
    first=np.append(first,ouch)

deito = []
for x in np.nditer(pred, op_flags=['readwrite']):
    x[...] += 733000
    r = dt.date.fromordinal(x)
    deito = np.append(deito, r)

predict=[]
for x in np.nditer(cutarr2, op_flags=['readwrite']):
    x[...] += 733000
    r = dt.date.fromordinal(x)
    predict = np.append(predict, r)

first=first.reshape(-1,1)
first=scaler.inverse_transform(first)
tes=scaler.inverse_transform(tes)
outarr2=scaler.inverse_transform(outarr2)


plt.plot(cutarr2,outarr2)
plt.plot(deito,first)
plt.plot(pred,tes)
plt.show()