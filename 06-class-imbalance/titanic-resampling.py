# -*- coding: utf-8 -*-
"""
Spyder Editor
Name:Itzu Huang
Topic:Resampling
"""
import pandas as pd
titanic=pd.read_csv("titanic-train(2).csv")
#to devide the data we could use the iloc
#[row,col],: => all, 0:-1 =>from first except last one
X=titanic.iloc[:,0:-1] #iloc 是依照位置選取資料
# : 表示所有列。
# 表示從第 1 欄開始，一直到最後一欄之前，不包含最後一欄。
y=titanic["Survived"]
#value_counts() means give number of category
print(y.value_counts())
#統計目標變數每個類別的筆數。
#表示未存活人數比存活人數多，資料存在類別不平衡。

'''======UnderSampling======'''
#Under_sampling will determine by the less category
#建立隨機欠抽樣工具。#random_state用來固定隨機結果，讓每次執行都抽到相同資料。
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(random_state=20260526)
X_resample,y_resample= rus.fit_resample(X,y)
#fit_resample() 會檢查 y 的類別數量，
#並把多數類別隨機刪除一些樣本，使它和少數類別一樣多。
#這就是undersampling
print(y_resample.value_counts())

'''======OverSampling======'''
#Over_sampling will determine by the larger category
from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=20260526)
X_resample2,y_resample2= ros.fit_resample(X,y)
#過抽樣不會刪除多數類別，而是隨機複製少數類別的資料，使兩類筆數相同。
print(y_resample2.value_counts())


'''======LabelEncoder======'''
#RandomUnderSampler 與 RandomOverSampler
#主要只是刪除或複製整列資料，所以原始資料中有文字時通常仍可處理。
#但是 SMOTE 需要計算樣本之間的距離，因此：
#所有自變數都必須是數值，且不能有缺失值。
#problem!!!! NOTE: every attribute have to be numeric
#it have to transform all the attributes as LableEncoder
#(ValueError: could not convert string to float: 'female')
#new_X.info() #Age 有missing value
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#建立 LabelEncoder。
#把文字類別轉成整數。
titanic["Sex"]=le.fit_transform(titanic["Sex"])
titanic["Cabin"]=le.fit_transform(titanic["Cabin"])
titanic["Embarked"]=le.fit_transform(titanic["Embarked"])

'''======missing value======'''
#檢查 Age 是否缺失。計算 Age 的中位數，並忽略 NaN。
import numpy as np
#preprocessing hte missing value
titanic["Age"]=np.where(titanic["Age"].isnull(),
                        np.nanmedian(titanic["Age"]),
                        titanic["Age"])

#將所有處理過的欄位重新組成自變數 DataFrame。
X2=pd.DataFrame([titanic["PassengerId"],titanic["Pclass"],
                titanic["Sex"],titanic["Age"],titanic["SibSp"],
                titanic["Parch"],titanic["Ticket"],
                titanic["Fare"],titanic["Cabin"],
                titanic["Embarked"]]).T

'''======SMOTE======'''
#SMOTE will setermine by the larger category
#but it will combine the less category data and larger category data
#最新的一定要會
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=20260526)
X_resample3,y_resample3= sm.fit_resample(X2,y)
print(y_resample3.value_counts()) #第1041個開始是合成的

#SMOTE 會針對少數類別進行合成抽樣。
#它不是單純複製原本的少數類別，而是：
# 1.找到某個少數類別樣本。
# 2.找到它附近的少數類別鄰居。
# 3.在兩個少數類別樣本之間產生新的合成樣本。

'''
讀取 Titanic
↓
將 Survived 設為 y
↓
查看類別數量
↓
RandomUnderSampler：刪除多數類別樣本
↓
RandomOverSampler：複製少數類別樣本
↓
處理文字與缺失值
↓
SMOTE：合成少數類別新樣本
↓
再次查看類別數量
'''
