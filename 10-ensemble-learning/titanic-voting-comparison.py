# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:54:17 2026
@author: Itzu Huang
@topic: DT,RF,KNN,SVM,voting(hard,soft)
"""
import pandas as pd

titanic=pd.read_csv("titanic-train(2).csv")
titanic.info()

titanic.drop(["PassengerId"], axis=1, inplace=True)
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Cabin"] = titanic["Cabin"].fillna(titanic["Cabin"].mode()[0])
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
sex = le.fit_transform(titanic["Sex"])

#we process region as oneHotencorder
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
cabin = ohe.fit_transform(titanic[["Cabin"]])
cabin = pd.DataFrame(cabin)
cabin.columns = ["Cabin_" + str(x)for x in ohe.categories_[0]]

embarked = ohe.fit_transform(
    titanic[["Embarked"]]
)

embarked = pd.DataFrame(embarked)
embarked.columns = ["Embarked_" + str(x)for x in ohe.categories_[0]]

X = pd.DataFrame([
    titanic["Pclass"],
    sex,
    titanic["Age"],
    titanic["SibSp"],
    titanic["Parch"],
    titanic["Fare"]]).T
X.columns = ["Pclass","Sex","Age",
             "SibSp","Parch","Fare"]
X = pd.concat([X, cabin, embarked],axis=1)

y=titanic["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test =train_test_split(X,y,
                                                  test_size=0.2,
                                                  random_state=20260602)
#========
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss.fit(X_train[["Age","Fare"]])
X_train[["Age","Fare"]] = ss.fit_transform(X_train[["Age","Fare"]])
#NOTE!!!!! the test data can't fit again, it just need to transform
X_test[["Age","Fare"]] = ss.transform(X_test[["Age","Fare"]])
X_train_std = X_train
X_test_std = X_test
#=========

from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(min_samples_leaf=2,
                          min_samples_split=0.05,
                          criterion="entropy",
                          random_state=20260602)
DT.fit(X_train_std,y_train)
print("The training ACC of DT=",DT.score(X_train_std, y_train))
print("The testing ACC of DT=",DT.score(X_test_std, y_test))

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=200, max_depth=8, random_state=20260602)
RF.fit(X_train_std,y_train)
print("The training ACC of RF=",RF.score(X_train_std, y_train))
print("The testing ACC of RF=",RF.score(X_test_std, y_test))

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train_std,y_train)
print("The training ACC of KNN=",KNN.score(X_train_std, y_train))
print("The testing ACC of KNN=",KNN.score(X_test_std, y_test))

#here we introduce the non-linear case as svc
from sklearn.svm import SVC
svm=SVC(gamma=0.1,C=1.0,kernel="rbf",probability=True, random_state=20260602)
svm.fit(X_train_std,y_train)
print("The training ACC of svm=",svm.score(X_train_std, y_train))
print("The testing ACC of svm=",svm.score(X_test_std, y_test))

from sklearn.ensemble import VotingClassifier
voting_hard=VotingClassifier(estimators=[("Decision Tree",DT),
                                         ("Random Forest",RF),
                                         ("Support Vectoe Machine",svm),
                                         ("K-nearest Neighbor",KNN)],
                             voting="hard",n_jobs=-1)
voting_hard.fit(X_train_std,y_train)
print("The training ACC of voting_hard=",voting_hard.score(X_train_std, y_train))
print("The testing ACC of voting_hard=",voting_hard.score(X_test_std, y_test))

voting_soft=VotingClassifier(estimators=[("Decision Tree",DT),
                                         ("Random Forest",RF),
                                         ("Support Vectoe Machine",svm),
                                         ("K-nearest Neighbor",KNN)],
                             voting="soft",n_jobs=-1)
voting_soft.fit(X_train_std,y_train)
print("The training ACC of voting_soft=",voting_soft.score(X_train_std, y_train))
print("The testing ACC of voting_soft=",voting_soft.score(X_test_std, y_test))
