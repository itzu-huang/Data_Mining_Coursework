# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:48:18 2026

@author:Itzu Huang
@topic: k-means in python
"""
import pandas as pd
titanic=pd.read_csv("titanic-train.csv")

titanic = titanic.drop(["Ticket"], axis=1)

import numpy as np
import statistics
titanic["Cabin"] = np.where(titanic["Cabin"].isnull(),
                            statistics.mode(titanic["Cabin"]),
                            titanic["Cabin"])

titanic["Embarked"] = np.where(titanic["Embarked"].isnull(),
                               statistics.mode(titanic["Embarked"]),
                               titanic["Embarked"])

titanic["Sex"] = np.where(titanic["Sex"].isnull(),
                          statistics.mode(titanic["Sex"]),
                          titanic["Sex"])

titanic["Age"] = np.where(titanic["Age"].isnull(),
                          np.nanmedian(titanic["Age"]),
                          titanic["Age"])

titanic["Fare"] = np.where(titanic["Fare"].isnull(),
                           np.nanmedian(titanic["Fare"]),
                           titanic["Fare"])


X = pd.DataFrame([titanic["Pclass"],titanic["Age"],
                  titanic["SibSp"],titanic["Parch"],
                  titanic["Fare"]]).T

X.columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)

sex = ohe.fit_transform(titanic[["Sex"]])
sex = pd.DataFrame(sex)
sex.columns = ohe.categories_[0]
cabin = ohe.fit_transform(titanic[["Cabin"]])
cabin = pd.DataFrame(cabin)
cabin.columns = ohe.categories_[0]
embarked = ohe.fit_transform(titanic[["Embarked"]])
embarked = pd.DataFrame(embarked)
embarked.columns = ohe.categories_[0]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic["Survived"] = le.fit_transform(titanic["Survived"])
y = titanic["Survived"]
X_new = pd.concat([X, sex, cabin, embarked],axis=1)

#第1題：看輪廓 → 可以加 Survived
#第2題：預測正確率 → 不可以加 Survived
#老師多次強調，標準化必須在資料分割 (Train-test split) 之後才做。
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_new,y, test_size=0.2,
                                                  random_state=20260609)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train)
X_train_std=ss.transform(X_train)
#NOTE!!!!! the test data can't fit again, it just need to transform
X_test_std=ss.transform(X_test)

#補欄位
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns).reset_index(drop=True)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns).reset_index(drop=True)
y_train_reset = pd.Series(y_train).reset_index(drop=True)

#SSE
SSE=[]
#第一題：使用目標變數看分群輪廓
X_train_with_y = pd.concat([X_train_std,
                            pd.Series(y_train_reset,
                                      name="Survived")],
                           axis=1)

from sklearn.cluster import KMeans
for i in range(10):
    #n-clusters=K, and i start from 0 hence need to +1
    kmeans=KMeans(n_clusters=i+1, init="k-means++",random_state=20260609)
    kmeans.fit(X_train_with_y)
    #centroid=inertia_
    SSE.append(kmeans.inertia_)
print(SSE)

import matplotlib.pyplot as plt

plt.plot(range(1,11),SSE,marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.title("Elbow Method")

#使用最佳K群進行kmeans(4,init="k-means++")
kmeans2=KMeans(n_clusters=4,init="k-means++",random_state=20260609)
kmeans2.fit(X_train_with_y)
print("SSE1=",kmeans2.inertia_) #SSE
#the final centroid=cluster_centers_
print(kmeans2.cluster_centers_)
#want to see the four cluster centroid
df=pd.DataFrame(kmeans2.cluster_centers_)
df.columns=X_train_with_y.columns

#silhouette_score 輪廓係數 higher means result better
from sklearn.metrics import silhouette_score
silhouette=silhouette_score(X_train_with_y,kmeans2.labels_,metric="euclidean")
print("silhouette1=",silhouette)

# 看各群 Survived 人數
print("Cluster vs Survived 人數")
print(pd.crosstab(kmeans2.labels_, y_train_reset))

# 看各群 Survived 比例
print("Cluster vs Survived 比例")
print(pd.crosstab(kmeans2.labels_, y_train_reset, normalize="index"))

#第二題：使用分群結果預測正確率
kmeans3=KMeans(n_clusters=4,init="k-means++",random_state=20260609)
kmeans3.fit(X_train_std)
print("SSE2=",kmeans3.inertia_) #SSE
print(kmeans3.cluster_centers_)
df=pd.DataFrame(kmeans3.cluster_centers_)
df.columns=X_train_std.columns

silhouette3=silhouette_score(X_train_std,kmeans3.labels_,metric="euclidean")
print("silhouette2=",silhouette3)

train_cluster = kmeans3.predict(X_train_std)
test_cluster = kmeans3.predict(X_test_std)
y_test_reset = pd.Series(y_test).reset_index(drop=True)
cluster_table = pd.crosstab(train_cluster, y_train_reset)
print("Train Cluster vs Survived")
print(cluster_table)

cluster_to_class = cluster_table.idxmax(axis=1).to_dict()
print("Cluster to Class")
print(cluster_to_class)


y_pred_test = pd.Series(test_cluster).map(cluster_to_class).to_numpy()
acc = (y_pred_test == y_test_reset).mean()
print("The cluster to predict classification ACC =", acc)

print("Test Survived vs Predicted Survived")
print(pd.crosstab(y_test_reset, y_pred_test))

#3.族群的輪廓與決策命名
profile_data = titanic.loc[X_train.index].copy()
profile_data = profile_data.reset_index(drop=True)
profile_data["Cluster"] = train_cluster

cluster_profile = profile_data.groupby("Cluster").agg(
    Count=("Survived", "count"),
    Survival_Rate=("Survived", lambda x: (x == 1).mean()),
    Age_Mean=("Age", "mean"),
    Fare_Mean=("Fare", "mean"),
    Pclass_Mean=("Pclass", "mean"),
    SibSp_Mean=("SibSp", "mean"),
    Parch_Mean=("Parch", "mean"),
    Sex_Mode=("Sex", lambda x: x.mode()[0]),
    Cabin_Mode=("Cabin", lambda x: x.mode()[0]),
    Embarked_Mode=("Embarked", lambda x: x.mode()[0])
    )

print(cluster_profile)
cluster_profile.to_csv("titanic_cluster_profile.csv", encoding="utf-8-sig")
