# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author:Itzu Huang
@topic: K-means
"""
import pandas as pd
bank=pd.read_csv("bank-data-clustering.csv")

'''=======資料處理========='''
'''=======LabelEncoder========='''

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
married=le.fit_transform(bank["married"])
children=le.fit_transform(bank["children"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])
#如果不做預測要加y的話
#bank["pep"]=le.fit_transform(bank["pep"])

'''=======OneHotEncoder========='''
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
region=ohe.fit_transform(bank[["region"]])
region=pd.DataFrame(region)
region.columns=ohe.categories_[0]

'''=======切資料========='''
''' 在這個case不用分割資料
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,
                                                  random_state=20260602)
因為就是分群不需要驗證
'''
#in this case because other attributes are boolean value, only age and income
#need to scale, so we use the scale directly to the attributes
#scale for only one attributem StandardScaler process more than two attributes
#StandardScaler suit for dataFrame
#老師多次強調，標準化必須在資料分割 (Train-test split) 之後才做。

'''=======scale (標準化)========='''
#因為除了age跟income其他在處理後已成為0或1的Boolean
#scale 處理單一屬性的簡單縮放，針對特定屬性直接縮放
from sklearn.preprocessing import scale

X=pd.DataFrame([scale(bank["age"]),sex,scale(bank["income"]),
                married,children,car,save_act,
                current_act,mortgage]).T
X.columns=["age","sex","income","married",
           "children","car","save_act","current_act","mortgage"]

X_new=pd.concat([X,region],axis=1)

y=bank["pep"]

'''=======KMeans ========='''
from sklearn.cluster import KMeans

#SSE
SSE=[]

for i in range(10):
    #n-clusters=K, and i start from 0 hence need to +1
    kmeans=KMeans(n_clusters=i+1, init="k-means++",random_state=20260609)
    kmeans.fit(X_new)
    #centroid=inertia_
    SSE.append(kmeans.inertia_)
print(SSE)

'''=======畫圖========='''
import matplotlib.pyplot as plt

plt.plot(range(1,11),SSE,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

'''=======給定分4群(n_clusters) 看結果========='''

kmeans2=KMeans(n_clusters=4,init="k-means++",random_state=20260609)
kmeans2.fit(X_new)
print(kmeans2.inertia_) #inertia_就是SSE
#the final centroid=cluster_centers_
print(kmeans2.cluster_centers_)
#want to see the four cluster centroid
df=pd.DataFrame(kmeans2.cluster_centers_)
df.columns=X_new.columns

#silhouette_score 輪廓係數 higher means result better
#−1 ≤ Silhouette Score ≤ 1
from sklearn.metrics import silhouette_score
silhouette=silhouette_score(X_new,kmeans2.labels_,metric="euclidean")
print("silhouette=",silhouette)
#輪廓係數會同時考慮：
#一筆資料與自己群內其他資料是否接近。
#一筆資料與最近的其他群是否相距夠遠。

#if we want to make the clusters to predict y
#用分群預測Y
X_pred=kmeans2.predict(X_new)
print(pd.crosstab(bank["pep"],X_pred))
print("the cluster to predict classficaiton acc=",(100+115+76+58)/600)

#沒有用y分群，預測會買不會買,預測跟實際比正確率
#如果不做預測就可以加入Y

#做期末報告的時候，如果不做預測的時候要加入y，如果不用預測的時候只用分群就不用抓y出來
