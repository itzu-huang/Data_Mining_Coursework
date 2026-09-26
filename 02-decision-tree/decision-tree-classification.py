# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:20:30 2026

@author:  Itzu Huang
@ Topic: Decision Tree (決策樹)
"""
import pandas as pd

bank=pd.read_csv("bank-no-missing.csv")
# for CART algorithm in python
#every attribute has to be numeric
#so need to use LabelEncode to transfer

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#打開資料集，除了PEP(目標變數)以外，其他沒有顏色的欄位都要做轉換
sex=le.fit_transform(bank["sex"]) #轉成0 1 的形式
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

X=pd.DataFrame([bank["age"],sex,region,bank["income"],
                married,bank["children"],car,
                save_act,current_act,mortgage]).T
#做一個新的X資料及，含轉換後的資料，and 不包含PEP
X.columns=["age","sex","region","income","married","children",
           "car","save_act","current_act","mortgage"]
y=bank["pep"]

#after the data preprocessing, we will devide the data into
#training dataset and test dataset

from sklearn.model_selection import train_test_split
#test_size = 0.2 means 80% train, 20% test
#random_state is for controlling all the case pick the same instances
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=20260324)


#前面的分成前面的,後面的分成後面的
 #import the CART Tree algorithm
 #min_samples_leaf=2 means whether the leaf are all the same class or not
 #when a node is lower than 2 instance then stop growing the tree

 #min_samples_split=0.2 means whether the leaf are all the same class or not
 #when a node percentage of different class is lower than 0.2 then stop growing
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(criterion="gini",min_samples_leaf=2, #一個節點只剩2個葉子就停
                           min_samples_split=0.2, random_state=20260324) #少數低於0.2就停

#use the model to fit training data
clf.fit(X_train,y_train)
#need to fit first then, use the score to the accuracy
print("The accuracy of training set = ",clf.score(X_train,y_train))
#when get the testing accuracy do not fit again, because it will change the model
#just fit the training and use the score to get the accuracy
print("The accuracy of training set = ",clf.score(X_test,y_test))

print("Leaves of hte tree = ", clf.get_n_leaves()) #列出葉子數
print("Depth of the tree = ", clf.get_depth()) #列出
