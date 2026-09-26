# -*- coding: utf-8 -*-
"""
Spyder Editor

@author:  Itzu Huang
@ topic: Miss value preprocessing, Descritization
"""
import pandas as pd

bank=pd.read_csv("missing-value.csv")
#survey the data wheather there is missong value?
#and the datatype(Dtype)
bank.info() #age，income，married，有missing value 599
#dropna()=> if you want to remove all the missing value obj
bank1=bank.dropna() #age, income, married 都有遺失值，所以刪除，並命名為bank1
#沒有人會一樣的，例如個人id學號、身分證之類的，可以刪除 #期中考要會，there is own number , identity
#drop(["column name"], axis=1, inplace=True)
#asix=1 means dalete the column
#axis=0 delete the row
#inplace=True => means directly replace the original DataFrame
bank.drop(["id"], axis=1, inplace=True) #bank 的size 變成(600,11)
bank.info()
#missing value attributes : age, married, income
#中位數補植 #numpy can do some easy statistics
import numpy as np

#nanmedian => find the median without consider the missing value 有遺失植下的中位數
print("the median of age",np.nanmedian(bank["age"]))
#nanmean => find the mean without cosider the missing value
print("the mean of age",np.nanmean(bank["income"]))
#numpy is not support the mode(眾數), the mode is for categorical data
#and find the which attribute value(屬性值) is the most frequent.
#Hence, we have to use to import statistics package and use the mode finction
import statistics

print("the mode of married=", statistics.mode(bank["married"]))

#want to replace the missing value by the result we found(mean,medain, mode)
#where function have three parameters
#first, bank["age"].isnull()
# => find where is missing value of the attribute "age" in bank
#Second, np.nanmedian(bank["age"]) =>
#Third, replace which column
bank["age"]=np.where(bank["age"].isnull(),
                     np.nanmedian(bank["age"]),
                     bank["age"]) #用median去取代遺失值

bank["income"]=np.where(bank["income"].isnull(),
                     np.nanmean(bank["income"]),
                     bank["income"]) #用mean去取代遺失值

bank["married"]=np.where(bank["married"].isnull(),
                     statistics.mode(bank["married"]),
                     bank["married"]) #用mode去取代遺失值

#Discretize the  numeric attribute into nominal 有顏色代表nominal
#age, income, children, is colorfull in python dataFrame

#demo bank["age"] cut into 3 parts

#this is to check the cutting points and the quantity of each part
#cut use bins=numbers
print(pd.cut(bank["age"],bins=3).value_counts()) #equalwidth 等距

#this is hte replace hte original column
bank["age"]=pd.cut(bank["age"],bins=3,
                   labels=["young","adult","mature"])

#This is to check the qcut points and the quantity of each part
#qcut use q=numbers
print(pd.qcut(bank["income"],q=3).value_counts()) #equalfreq 等百分比
#this is hte replace the original column
bank["income"]=pd.qcut(bank["income"],q=3,labels=["L","M","H"])

#bank["children"].astype(str) => transform the numberic value directly into naminal 把數字型態變成文字型態
bank["children"]=bank["children"].astype(str) #因為children 是 0 1 2 3 去分類所以要用astype

bank.info()
#transfer the dataFrame to csv file

#use the to_csv function to transfer the DFinto csv
#bank.to_csv("file name", index=False means )
bank.to_csv("productsales-preprocessed.csv",index=False)
