# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
@author:Itzu Huang
@topic: select attributes in python
"""
import pandas as pd
#read the file
bank=pd.read_csv("bank-no-missing.csv")
#cheke the missing value and the data type
bank.info()

#use the label Encoder to do the preprocessing
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sex=le.fit_transform(bank["sex"])
region=le.fit_transform(bank["region"])
married=le.fit_transform(bank["married"])
car=le.fit_transform(bank["car"])
save_act=le.fit_transform(bank["save_act"])
current_act=le.fit_transform(bank["current_act"])
mortgage=le.fit_transform(bank["mortgage"])

#rebuild the X and transpose
X=pd.DataFrame([bank["age"],sex,region,bank["income"],
                married,bank["children"],car,save_act,
                current_act,mortgage]).T

#gice the X columns name
X.columns=["age","sex","region","income","married",
           "children","car","save_act","current_act","mortgage"]

#define the y variable
y=bank["pep"]

#split the data into train and test the percentage is 20% test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2,
                                                  random_state=20260421)
#Build the training model
from sklearn.tree import DecisionTreeClassifier
clf= DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,
                            min_samples_split=0.01,random_state=20260421)
#配適建模資料集 fit the training model
clf.fit(X_train,y_train)

#用clf.score 列出正確率 usr the clf.score to find the accuracy
print("train Acc=",clf.score(X_train,y_train))
print("test Acc=",clf.score(X_test,y_test))
#to get the leaves
print("Leaves of the tree=", clf.get_n_leaves())
#to get the depth of the tree
print("Depth of the tree=", clf.get_depth())


#to build the text tree(文字樹)
from sklearn.tree import export_text
tree=export_text(clf, feature_names=list(X.columns))
print(tree)


#====以下為新打的=====

#build the tree with entropy
clf2= DecisionTreeClassifier(criterion="entropy",min_samples_leaf=2,
                            min_samples_split=0.01,random_state=20260421)

#配適建模資料集 fit hte training model
clf2.fit(X_train,y_train)

#用clf2.score 列出正確率 use the clf2.score to find the accuracy
print("Entropy train Acc=",clf2.score(X_train,y_train))
print("Entropy test Acc=",clf2.score(X_test,y_test))
#to get the leaves
print("Entropy Leaves of the tree=", clf2.get_n_leaves())
#to get the depth of the tree
print("Entropy Depth of the tree=", clf2.get_depth())

#print the better tree of gini  and Entropy based on the test acc
#to build the text  tree(文字樹)
from sklearn.tree import export_text
tree=export_text(clf2, feature_names=list(X.columns))
print(tree)


#select Attributes by Chi-square method 用卡方找最強的
#we need two function (1)selectbest , (2) chi2
# 需要使用兩個工具：
# (1) SelectKBest：選出分數最高的前 K 個屬性
# (2) chi2：使用卡方檢定作為評分方法
#卡方
from sklearn.feature_selection import SelectKBest,chi2
# 建立卡方特徵選擇器，選出前 5 個最佳屬性
sk=SelectKBest(chi2,k=5)
#use the sk to fit the training dataset
sk.fit(X_train, y_train)

#print the best 5 attribute by chi-square
print("the best 5 attributes by chi-square=",sk.get_feature_names_out())

#select the top 5 as a new file
X_new_train=sk.transform(X_train)

#transfer the file as pd DataFrame
X_new_DF=pd.DataFrame(X_new_train)

#give the  column name
X_new_DF.columns=['age','income','married','children','save_act']

#then we rebuild the model by top 5 chi-square attributes
# 使用卡方選出的前 5 個屬性重新建立決策樹模型
from sklearn.tree import DecisionTreeClassifier
clf3= DecisionTreeClassifier(criterion="entropy",min_samples_leaf=2,
                            min_samples_split=0.01,random_state=20260421)

#use the top 5 data to fit the modle
clf3.fit(X_new_DF,y_train)
#the training ACC of chi-square top 5 model
print("the training ACC of chi-square top 5=",clf3.score(X_new_DF, y_train))
#to find the importance of the top 5 attributes based on chi-square
# 印出卡方前 5 個屬性模型中各屬性的重要性
print("the attribue importance of chi-square top 5=",clf3.feature_importances_)
#we can't directly get the feature importance , hence we have to
#build the model first, then get the feature importance
#the Test ACC of the ACC of chi-square top 5=
# 無法只靠卡方直接取得決策樹的屬性重要性
# 必須先建立並訓練模型，才能取得 feature_importances_

X_new_test=sk.transform(X_test)

#transfer the file as pd DataFrame
X_new_test=pd.DataFrame(X_new_test)

#give the  column name
X_new_test.columns=['age','income','married','children','save_act']

print("the testing ACC of chi-square top 5=",clf3.score(X_new_test, y_test))


#第二個方法selectfrommodel
#now we use the same model to find the top 5 attribute
clf3.fit(X_train,y_train)
print("use all the attribute to find the feature importance")
print(clf3.feature_importances_)

#built a "a" array to record the feature_importance_
a=clf3.feature_importances_
#combine the X.columns and a to  build a datafram
importance_df=pd.DataFrame([X.columns,a]).T
#rank the importance by click the datafram
#the importance now are unfair because it's based on 10 attributes
#hence we have to rebuild a 5 attribute dataframe for model feature selection method

X_train_5=X_train[["income","children","age","mortgage","save_act"]]
#use the clf3 to fit tje train data and y_train to build the mod
clf3.fit(X_train_5,y_train)
#create the testing dataset for model select top 5
X_test_5=X_test[["income","children","age","mortgage","save_act"]]

#the testing ACC of model feature selection top 5
print("the training ACC of model feature selection top 5=",clf3.score(X_train_5,y_train))

print("the testing ACC of model feature selection top 5=",clf3.score(X_test_5,y_test))



#because this time the clf3 is fit the model selection top 5
#hence this time only will comput
print("use all the attribute to find the feature importance")
print(clf3.feature_importances_)


# 因為卡方方法的 test accuracy 比模型特徵選擇法更好
# 所以選擇卡方方法選出的前 5 個屬性來建立文字版決策樹
#chi square 的test Acc is more better so ues chi square to build the tree 卡方比selectfrommodel好，所以選卡方
#chose the better top 5 to build the tree
clf3.fit(X_new_DF, y_train)

#print the better tree of "Entropy" based on the test acc
#to build the text tree(文字樹)
from sklearn.tree import export_text
#用clf3的建模 卡放最強的五個的資料去建文字樹
tree=export_text(clf3, feature_names=list(X_new_DF.columns))
print(tree)

#to download the graphviz first
#IMPORTANT!!!!!!!! please not chooose "do not" for the users
#At least choose for the current user
#go to anaconda promppt > pip install graphviz

#because we need to export the tree by sklean import tree package
from sklearn import tree


import graphviz
#tree.export_graphviz(model, feature_names,class_names,proportion)
#set all the parameters here have assign to tree_graph
tree_graph=tree.export_graphviz(clf3, feature_names=X_new_DF.columns,
                                   class_names=clf3.classes_,proportion=True,rounded=True)

graph_chi2_top5=graphviz.Source(tree_graph)

#give hte file format type as "png" to the graph

graph_chi2_top5.format="png"
graph_chi2_top5.render("feature-selection-tree",view=False)












'''
#print the better tree of gini and ENtropy based on the test acc,
#to build the text tree(文字樹)
from sklearn.tree import export_text
tree2=export_text(clf2, feature_names=list(X.columns))
print(tree2)

'''
