# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author:Itzu Huang
@topic: Random Forest
"""
import pandas as pd
bank=pd.read_csv("bank-no-missing.csv")
#bank.info()

'''=======資料處理========='''
'''=======LabelEncoder========='''
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
#region is not LabelEncoder have to OneHotEncoder
#region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

X=pd.DataFrame([bank["age"],sex,bank["income"],
                married,bank["children"],car,save_act,
                current_act,mortgage]).T
X.columns=["age","sex","income","married",
           "children","car","save_act","current_act","mortgage"]

'''=======OneHotEncoder========='''
#we process region as oneHotencorder
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

#use concat to combine multiple dataframe
X_new=pd.concat([X,region],axis=1)

#tree series(Decision Treem Random Forest) do not need to SS
#also do not need to transform as numeric

y=bank["pep"]

'''=======切資料========='''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,
                                                  random_state=20260602)

'''=======RandomForest========='''
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=200, max_depth=8,
                          random_state=20260602)
#n_estimators=200 表示建立： 200 棵決策樹。
#樹越多通常結果越穩定，但運算時間也會增加。
#max_depth=8 表示每棵決策樹的最大深度為 8。
#限制深度可以避免單棵樹長得太複雜，降低overfitting的風險

#配適建模資料集
RF.fit(X_train,y_train)
#use RF.score to find the accuracy
print("train Acc Of RF=",RF.score(X_train,y_train))
print("test Acc Of RF=",RF.score(X_test,y_test))
#兩者差距很大，可能代表overfitting。
#差距較小，模型的泛化能力通常比較合理。
