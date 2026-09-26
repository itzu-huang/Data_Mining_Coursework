# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author:Itzu Huang
@topic: Overfitting and parameters tuning + cross validaiton
"""
import pandas as pd


bank=pd.read_csv("bank-no-missing.csv")
#bank.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

X=pd.DataFrame([bank["age"],sex,region,bank["income"],
                married,bank["children"],car,save_act,
                current_act,mortgage]).T
X.columns=["age","sex","region","income","married",
           "children","car","save_act","current_act","mortgage"]

y=bank["pep"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,
                                                  random_state=20260505)


from sklearn.tree import DecisionTreeClassifier


#min_sample_leaf=2 >? 必考 min_samples_split=0.2 =>? 少數類別小於0.2就會停止
#meas when the value larger means the tree will be simpler
#while the value smaller means the tree will be complex and bigger
#remenber hte decision tree also need random_state

#create an empty list acc
acc=[]

#we use the i from 10 to 1
#we fixed the min_samples_leaf=2 because it's most strictly
for i in range(10,0,-1):
    clf= DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                            min_samples_split=i/100,random_state=20260505)
    #every time we fit X_train and y_train for the model
    clf.fit(X_train,y_train)
    #use the clf.score(X_train,y_train) to get acc
    #print("train Acc=",clf.score(X_train,y_train))
    #print("test Acc=",clf.score(X_test,y_test))
    #print("Leaves of the tree=", clf.get_n_leaves())
    #print("Depth of the tree=", clf.get_depth())
    acc.append([i/100,clf.score(X_train,y_train),clf.score(X_test,y_test),
               clf.get_n_leaves(),clf.get_depth()])
#because the acc is a list not easy to observe, hense transfer to DataFrame

data=pd.DataFrame(acc)
data.columns=["split","train Acc","test Acc","Leaves of the tree","Depth of the tree"]

#cross validaiton have to  import the package
from sklearn.model_selection import cross_val_score
#use the best aprameters based on the previous part
#min_samples_split=0.03 which is before overfitting
clf1=DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                        min_samples_split=0.03,random_state=20260505)

#cross-validation fold=0
#cross_val_score(model,full X,full y,cv=fold,scoring="accuracy")
cross=cross_val_score(clf1,X,y,cv=10,scoring="accuracy")
#cross.mean()=> average all the results # cv = fold = 10 => 代表做10次
print("full data cross-validation fold=10, ACC=",format(cross.mean(),".4f"))
