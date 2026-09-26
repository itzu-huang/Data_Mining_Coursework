# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:12:51 2026

@author:Itzu Huang
Topic:One-Hot Encoding & KNN
"""
import pandas as pd
bank=pd.read_csv ("bank-no-missing.csv")

'''=======資料處理========='''
'''=======OneHotEncoder========='''
from sklearn.preprocessing import OneHotEncoder #沒有排序的欄位
ohe=OneHotEncoder(sparse_output=False)
#different from labelEncoder here have to double[]
sex=ohe.fit_transform(bank[["sex"]])
#but now it's array, we need the dataFrame
sex=pd.DataFrame(sex)
#but without the columns name
sex.columns=ohe.categories_[0]

region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

married=ohe.fit_transform(bank[["married"]])
married=pd.DataFrame(married)
#trace method ["married_" + s for s in something] 每一個欄位分開後都是yes no
married.columns=["married_"+s for s in ohe.categories_[0]]
#boolin值要加這行 ， 因為布林值的問題yes no的欄位重複

car=ohe.fit_transform(bank[["car"]])
car=pd.DataFrame(car)
car.columns=["car_"+s for s in ohe.categories_[0]]
#boolin值要加這行 ， 因為布林值的問題yes no的欄位重複

save_act=ohe.fit_transform(bank[["save_act"]])
save_act=pd.DataFrame(save_act)
save_act.columns=["save_"+s for s in ohe.categories_[0]]
#boolin值要加這行 ， 為了避免欄位名稱重複，讓欄位意義更清楚。

current_act=ohe.fit_transform(bank[["current_act"]])
current_act=pd.DataFrame(current_act)
current_act.columns=["current_"+s for s in ohe.categories_[0]]
#boolin值要加這行 ， 因為布林值的問題yes no的欄位重複

mortgage=ohe.fit_transform(bank[["mortgage"]])
mortgage=pd.DataFrame(mortgage)
mortgage.columns=["mortgage_"+s for s in ohe.categories_[0]]
#boolin值要加這行 ， 因為布林值的問題yes no的欄位重複

'''=======訂定資料集========='''
X=pd.concat([bank["age"],sex,region,bank["income"],
             married,bank["children"],car,save_act,
             current_act,mortgage],axis=1)

#KNN the target variable do not need to transform to numeric
#can use the original one
y=bank["pep"]

'''=======切資料========='''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=20260526)
#[NOTE!!! APPEAR in Final EXAM]會考 split之後才標稕化的原因
#because the testing set represent the population, it's unknown
#when we do the scaler it shouldn't be count in
#we just can use the training set to do the scaler
#and make the testing set to fit and transform by the training set
#KNN 是使用距離判斷鄰居，因此不同欄位的尺度會直接影響距離。
#若不標準化，income 可能主導距離計算。
#只能使用訓練集算出的平均數與標準差，不可重新 fit。
#原因是測試集模擬未知資料，不能讓它的資訊提前進入前處理過程，否則會造成資料洩漏。

'''=======StandardScaler========='''
#但train & test 都要StandardScaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() #標準化原本就是數值的欄位

X_train[["age"]]=ss.fit_transform(X_train[["age"]]) #標準化age的數值
#for testing data, just transform not fit anymore!!!
X_test[["age"]]=ss.transform(X_test[["age"]])

X_train[["income"]]=ss.fit_transform(X_train[["income"]]) #標準化age的數值
#for testing data, just transform not fit anymore!!!
X_test[["income"]]=ss.transform(X_test[["income"]])

X_train[["children"]]=ss.fit_transform(X_train[["children"]]) #標準化age的數值
#for testing data, just transform not fit anymore!!!
X_test[["children"]]=ss.transform(X_test[["children"]])

#測試集代表未知的母體。我們只能用訓練集的平均數和標準差來 fit 標準
#再應用到測試集。若混在一起算，會造成資訊洩漏 (Data Leakage)
#這是不公平且錯誤的實驗方式

'''=======KNN 執行========='''
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
#KNN here is useless, the training set will find itself as the nearset
#neightbor, hence the training ACC will higher estimate
#hence, it hust use for find the best K
knn.fit(X_train,y_train)
print("KNN training ACC=",knn.score(X_train,y_train)) #LAZY KNN
#註解說訓練集會找到自己作為最近鄰，這個方向是對的，
#尤其當 K 很小時，訓練正確率通常偏高。
#但不能說訓練正確率「沒有用」，比較精確的說法是：
#KNN 的訓練正確率通常較樂觀，不能單獨用來選擇最佳 K，
#仍應觀察驗證集或測試集表現。


#use the loop to find the best K
acc=[]

for i in range(1,481):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    #X_train have scaler but not X_test
    #print("K=",i,"Test ACC=", knn.score(X_test,y_test))
    acc.append(knn.score(X_test,y_test))

print("The MAX ACC=",max(acc))

bestK=0
for i in range(0,480):
    if acc[i]==max(acc):
        bestK=i+1

print("The best K=", bestK)
