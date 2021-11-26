# -*- coding: utf-8 -*-
# @File : code.py
# @Author : HK560
# @Time : 2021/11/11 17:58:45
# @LastUpdate :2021/11/18 19:14:21

from matplotlib import colors
from matplotlib.text import Annotation
from numpy.lib.shape_base import apply_along_axis, column_stack
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from math import sqrt
import operator
from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# 读取数据,预处理
data = pd.read_csv('housing.csv') 
#去掉含有nan的行
data=data.dropna(axis=0,how='any')
#独热编码
OneHot_data=pd.get_dummies(data[['ocean_proximity']])
data=data.join(OneHot_data)
data=data.drop(['ocean_proximity'],axis=1)

#归一化
data=data/data.max(axis=0) 
#print(data)

#划分训练集和测试集
train_data=data.sample(frac=0.7, random_state=0,axis=0)
test_data=data[~data.index.isin(train_data.index)]
#导出测试数据训练数据的csv文件
train_data.to_csv("train_data.csv",index=False)
test_data.to_csv("test_data.csv",index=False)
#取出X和Y
# data_org_Y=data_org[['median_house_value']]
# data_org_X=data_org.drop(['median_house_value'],axis=1)
test_data_Y=test_data[['median_house_value']]
test_data_X=test_data.drop(['median_house_value'],axis=1)
train_data_Y=train_data[['median_house_value']]#取出Y
train_data_X=train_data.drop(['median_house_value'],axis=1)

# test_data_Y_max=test_data_Y.max(axis=0)
# test_data_X_max=test_data_X.max(axis=0)
# train_data_Y_max=train_data_Y.max(axis=0)
# train_data_X_max=train_data_X.max(axis=0)
# 
# test_data_Y=test_data_Y/test_data_Y_max
# test_data_X=test_data_X/test_data_X_max
# train_data_Y=train_data_Y/train_data_Y_max
# train_data_X=train_data_X/train_data_X_max

#导出训练集的X和Y
train_data_X.to_csv("train_data_X.csv",index=False)
train_data_Y.to_csv("train_data_Y.csv",index=False)

# Y_org=data_org_Y.values
# X_org=data_org_X.values

X_test=test_data_X.values
Y_test=test_data_Y.values
X=train_data_X.values
Y=train_data_Y.values
print(X.shape)
print (Y.shape)
#梯度下降
W=np.zeros(shape=[1,X.shape[1]])
b=0
#设置学习速率
alpha= 0.001
#学习次数
learning_count =100000
for i in range(learning_count):
    #y_h 是预测的y值
    y_h=np.dot(X,W.T) + b
    #求出损失
    lost=y_h-Y
    W=W - alpha*(1/len(X))*np.dot(lost.T,X)
    #给欧米噶（其实就是theta）做梯度下降
    b=b-alpha*(1/len(X))* lost.sum()
    #代价
    cost = (lost**2)/(len(X))
    if i%10000==0:
        print ('梯度下降中...')

#求出闭合式的theta
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


def R2(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)


print('测试集Y值:')
print(Y_test)

print ('——————线性回归梯度下降——————')
print('theta系数:')
print(W)
print('b值：')
print (b)
y_pre= np.dot(X_test,W.T)+b
print('测试集预测值：')
print(y_pre)

mse_test = np.sum((y_pre - Y_test) ** 2) / len(Y_test)
print('MSE:%e'%mse_test)
rmse_test = sqrt(mse_test)
print('RMSE:%e'%rmse_test)
mae_test = np.sum(np.absolute(y_pre - Y_test)) / len(Y_test)
print('MAE:%e'%mae_test)
r2_s=R2(Y_test,y_pre)
print('R2:%e'%r2_s)

print ('——————线性回归闭式解——————')
print('theta系数:')
print(theta)
y_predict = X_test.dot(theta)
print('测试集预测值：')
print(y_predict)

mse_test = np.sum((y_predict - Y_test) ** 2) / len(Y_test)
print('MSE:%e'%mse_test)
rmse_test = sqrt(mse_test)
print('RMSE:%e'%rmse_test)
mae_test = np.sum(np.absolute(y_predict - Y_test)) / len(Y_test)
print('MAE:%e'%mae_test)
r2_s=R2(Y_test,y_predict)
print('R2:%e'%r2_s)



plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.figure(1)
plt.title("longitude")
pic_X=test_data_X[['longitude']]
#pic_pred_Y=reduce(operator.add, y_predict)
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(2)
plt.title("latitude")
pic_X=test_data_X[['latitude']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(3)
plt.title("housing_median_age")
pic_X=test_data_X[['housing_median_age']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(4)
plt.title("total_rooms")
pic_X=test_data_X[['total_rooms']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(5)
plt.title("total_bedrooms")
pic_X=test_data_X[['total_bedrooms']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(6)
plt.title("population")
pic_X=test_data_X[['population']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(7)
plt.title("households")
pic_X=test_data_X[['households']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(8)
plt.title("median_income")
pic_X=test_data_X[['households']]
plt.scatter(pic_X,Y_test,color='blue')
plt.scatter(pic_X,y_predict,color='red')

plt.figure(9)
plt.title("闭式解预测房价")
plt.scatter(Y_test,y_predict,color='red')
plt.plot(Y_test,Y_test,color='blue')

plt.figure(10)
plt.title("梯度下降预测房价")
plt.scatter(Y_test,y_pre,color='red')
plt.plot(Y_test,Y_test,color='blue')

# plt.figure(11)
# plt.title("org")
# plt.scatter(Y_test*train_data_Y_max,y_pre*train_data_Y_max,color='red')
# plt.plot(Y_test*train_data_Y_max,Y_test*train_data_Y_max,color='blue')

plt.grid()
plt.show()