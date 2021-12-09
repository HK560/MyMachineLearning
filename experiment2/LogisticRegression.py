# -*- coding: utf-8 -*-
# @File : LogisticRegression.py
# @Author : HK560
# @Time : 2021/12/09 15:39:08
import math
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


print("逻辑回归分类")

# 读取数据,预处理
df = pd.read_csv('adult.csv') 
#去掉含有nan的行
df=df.dropna(axis=0,how='any')
#预处理，去掉缺省属性的行
df=df[~df['workclass'].isin(["?"])]
df=df[~df['native-country'].isin(["?"])]
df=df[~df['occupation'].isin(["?"])]
#将结果转为二分类值为0 1方便后续处理。重命名为target列
df.loc[df['income'] == ">50K","target"] = 1
df.loc[df['income'] != ">50K","target"] = 0
df.drop(['income'], axis=1 ,inplace=True)

#选出离散型特征
cat_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 
               'gender','native-country']
#连续性特征
num_columns = ['age','fnlwgt','educational-num','capital-gain', 'capital-loss', 'hours-per-week']
#目标特征
target_column = "target"
#对离散型特征独热编码
encoded_df = pd.get_dummies(df,columns=cat_columns)

#特征值
df_x = encoded_df.drop(columns="target")
#目标值
df_y = encoded_df["target"].values

#将连续型特征值归一化
num_mean = df_x[num_columns].mean()
num_std = df_x[num_columns].std()
num_normol = (df_x[num_columns] - num_mean)/num_std
df_x.drop(columns=num_columns,inplace=True)
df_x = pd.concat([df_x,num_normol],axis=1).values   

#划分训练集测试集
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=2,train_size=0.7)
for train_index,test_index in sss.split(df_x,df_y):
    trainx,testx = df_x[train_index],df_x[test_index]
    trainy,testy = df_y[train_index],df_y[test_index]


#sigmoid函数
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

X=trainx
Y=np.reshape(trainy, (-1, 1))
X_test=testx
Y_test=np.reshape(testy, (-1, 1))
# print(X.shape)
# print (Y.shape)

theta=np.zeros(shape=[1,X.shape[1]])
b=0
#设置学习速率
alpha= 0.001
#学习次数
learning_count =100
theta = np.zeros(shape=[1, X.shape[1]])
theta = theta.T
for i in range(learning_count):
    pre = np.dot(X, theta)
    w = 1 / (1 + np.exp(-pre))  
    h = np.dot(X.T, (Y - w))  
    h = h / len(h)
    lost = -(np.dot(Y.T, np.log(w)) + np.dot((1 - Y).T, np.log(1 - w))) / len(Y)
    theta = theta + alpha * h
    if i%100==0:
      print ('梯度下降中...')

y_pre= np.dot(X_test,theta)

y_preSigmd=sigmoid(y_pre)
#将预测出的值划分为0 1
y_preSigmd= np.where(y_preSigmd>0.5,1,0)



# df.to_csv("datanotnan.csv",index=False)   

print(y_preSigmd)

#打印损失函数
print("损失函数\n",lost)


#计算准确率
from sklearn.metrics import accuracy_score
print("准确率：",accuracy_score(y_preSigmd, Y_test))
# print(ac/all)


#计算auc值
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(Y_test,y_preSigmd)
print("AUC值:",auc_score)


#Reference
#逻辑回归模型预测成年人收入水平 https://zhuanlan.zhihu.com/p/45835659
