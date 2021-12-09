# -*- coding: utf-8 -*-
# @File : NaiveBayes.py
# @Author : HK560
# @Time : 2021/12/09 15:38:56
from os import error
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
 
print("朴素贝叶斯分类")

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
df.drop(['educational-num'], axis=1,inplace=True)

#将连续型特征划分区间
df['age']=pd.cut(df['age'],bins=10,right=True)
df['capital-gain']=pd.cut(df['capital-gain'],bins=3,right=True)
df['capital-loss']=pd.cut(df['capital-loss'],bins=3,right=True)
df['hours-per-week']=pd.cut(df['hours-per-week'],bins=20,right=True)
df['fnlwgt']=pd.cut(df['fnlwgt'],bins=20,right=True)
#划分训练集和测试集
train_data=df.sample(frac=0.7, random_state=0,axis=0)
test_data=df[~df.index.isin(train_data.index)]

#因为测试集太多，预测计算需要较长时间，这里只取一部分
test_data=test_data[0:100]

df_x = df.drop(columns="target")
df_y = df["target"].values


trainX=train_data.drop(columns="target")
trainY=train_data['target']
testX=test_data.drop(columns="target")
testY=test_data['target']

df.to_csv("datanotnanBeiyesi.csv",index=False)   
class NaiveBayes(object):
    def __init__(self, X_train, y_train):
        self.X_train = X_train  #样本特征
        self.y_train = y_train  #样本类别
        #训练集样本中每个类别(二分类)的占比，即P(类别)，供后续使用
        self.P_label = {1: np.mean(y_train.values), 0: 1-np.mean(y_train.values)}
 
    #在数据集data中, 特征feature的值为value的样本所占比例
    #用于计算P(特征|类别)、P(特征)
    def getFrequency(self, data, feature, value):
        num = len(data[data[feature]==value]) #个数
        return num / (len(data))
 
    def predict(self, X_test):
        self.prediction = [] #预测类别
        # 遍历样本
        for i in range(len(X_test)):
            x = X_test.iloc[i]      # 第i个样本
            P_feature_label0 = 1    # P(特征|类别0)之和
            P_feature_label1 = 1    # P(特征|类别1)之和
            P_feature = 1           # P(特征)之和
            # 遍历特征
            for feature in X_test.columns:
                # 分子项，P(特征|类别)
                data0 = self.X_train[self.y_train.values==0]  #取类别为0的样本
                P_feature_label0 *= self.getFrequency(data0, feature, x[feature]) #计算P(feature|0)
 
                data1 = self.X_train[self.y_train.values==1]  #取类别为1的样本
                P_feature_label1 *= self.getFrequency(data1, feature, x[feature]) #计算P(feature|1)
 
                # 分母项，P(特征)
                P_feature *= self.getFrequency(self.X_train, feature, x[feature])
 
            #属于每个类别的概率
            if i%25==0:
                # print ('...')
                print("预测第",i,"个")
                print(P_feature_label0)
                print(P_feature_label1)
            P_0 = (P_feature_label0*self.P_label[0]) / P_feature
            P_1 = (P_feature_label1 * self.P_label[1]) / P_feature
            #选出大概率值对应的类别
            self.prediction.append([1 if P_1>=P_0 else 0])
        return self.prediction


model = NaiveBayes(trainX, trainY)    #训练
# print(X_test.shape)
y_pre = model.predict(testX)    
print(y_pre)       

#准确率
from sklearn.metrics import accuracy_score
print("准确率：",accuracy_score(y_pre, testY))
#auc值
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(testY,y_pre)
print("AUC值:",auc_score)

#Reference
# 朴素贝叶斯代码_《机器学习》之 朴素贝叶斯原理及代码 https://blog.csdn.net/weixin_39710249/article/details/111373540