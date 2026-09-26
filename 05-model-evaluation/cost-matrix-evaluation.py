# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author:Itzu Huang
@topic: cost Matrix
"""
import pandas as pd
#讀取資料
bank=pd.read_csv("bank-no-missing.csv")
bank.info()
#類別變數轉成數字
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

#建立自變數 X
X=pd.DataFrame([bank["age"],sex,region,bank["income"],
                married,bank["children"],car,save_act,
                current_act,mortgage]).T
X.columns=["age","sex","region","income","married",
           "children","car","save_act","current_act","mortgage"]
#建立目標變數 y
y=bank["pep"]

#分割訓練集與測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,
                                                  random_state=20260519)

#建立決策樹模型
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                            min_samples_split=0.2,
                            random_state=20260519)
#配適建模資料集，訓練模型
clf.fit(X_train,y_train)
#用clf.score 列出正確率
print("train Acc=",clf.score(X_train,y_train))
print("test Acc=",clf.score(X_test,y_test))
#查看決策樹大小
print("Leaves of the tree=", clf.get_n_leaves())
print("Depth of the tree=", clf.get_depth())

'''==========='''
#If we need the confusion, we need to predict X 預測結果
y_pred=clf.predict(X)
#print the cm(confusion matrix) => Real first, Presict later
#pd.crosstab => use the pandas to built the crossrable
#混淆矩陣 confusion matrix
#講中文很重要：真實的放左邊、預測的放右邊，但在真實的裡面yes跟no可以交換
#但不管怎麼樣都是看對角線
cm=pd.crosstab(y,y_pred)
print()
print(cm)
#(Rows) 必須是 Actual Class (真實類別)
#(Columns) 必須是 Predicted Class (預測類別)
#Total Accuracy
print("total ACC=",(280+194)/600)
print("Precision=",194/(467+194))
#Precision =精確率 ,in the case, guess less but target the goal
print("Recall=",194/(80+194))
#Recall=喚回率, in the case to find out all the answer is important
#in this case recall is higher
print("F-measaure",(194*2)/(194*2+46+80)) #F-measure = F-score =F1-score
#the Best case is Precision and Recall=1
#means all the guess totally target all the answers
print()
#在稀有事件 (Rare Event) 中，正確率會具有誤導性
'''========='''
#加入cost matrix
import numpy as np
cost=np.array([[0,1],[100,-1]]) #matrix in ppt to trans
print("Total cost=",np.multiply(cm,cost).sum().sum())
