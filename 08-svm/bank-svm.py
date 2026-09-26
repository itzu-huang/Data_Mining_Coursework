# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Itzu Huang
"""
import pandas as pd

bank=pd.read_csv("bank-no-missing.csv")
#bank.info()
#because SVM must all numberic attributes
#LabelEncorder or OneHotEncorder
#today we demo the subset 子集合
# | vertical bar represent or (這個觀念很重要) , | 代表「或 OR」。
subdata=bank[(bank["region"]=="INNER_CITY")|
             (bank["region"]=="TOWN")|
             (bank["region"]=="RURAL")]
#region 等於 INNER_CITY，或等於 TOWN，或等於 RURAL，就保留該筆資料。

#use the age, income and children to classify
#the region of the customor lived
X=subdata[["age","income","children"]]
#模型會利用這三個特徵來預測客戶住在哪一種地區。

#SVM ask the target variable need to numeric
#but is just because python package,not for the method
y=subdata[["region"]]

'''=======轉成數值資料========='''
#SVM must all numberic attributes 使用 LabelEncoder。
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

'''=======切資料========='''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test =train_test_split(X,y,
                                                  test_size=0.2,
                                                  random_state=20260602)

'''=======StandardScaler========='''
#因為SVM 會利用變數之間的距離與超平面來分類。
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train_std=ss.transform(X_train)
#NOTE!!!!! the test data can't fit again, it just need to transform
X_test_std=ss.transform(X_test)

'''=======LinearSVC========='''
from sklearn.svm import LinearSVC
m=LinearSVC(C=0.5,dual=False,class_weight="balanced")
#dual=False 表示 LinearSVC 可以解原始問題或對偶問題。
m.fit(X_train_std,y_train) #訓練 SVM 模型
print("The training accuracy of SVM=", m.score(X_train_std, y_train))
print("The testing accuracy of SVM=", m.score(X_test_std, y_test))

'''=======f1-score========='''
#f1 score in the sklern.metrics package
#because f1_score need to compare the real data and predict data
#hence have to predict first
from sklearn.metrics import f1_score
y_pred_test=m.predict(X_test_std)
print("number of wrong classification", (y_test!=y_pred_test).sum())
#逐筆比較真實值與預測值。若預測錯誤會得到 True，預測正確會得到 False。
#計算錯分筆數

y_pred_train=m.predict(X_train_std)
print("number of wrong classification =", (y_train!=y_pred_train).sum())
print("F1-score for training =", f1_score(y_train, y_pred_train,average="weighted"))
print("F1-score for testing =", f1_score(y_test, y_pred_test,average="weighted"))
#F1-score 是 Precision 與 Recall 的調和平均，數值越接近 1，模型表現通常越好。
