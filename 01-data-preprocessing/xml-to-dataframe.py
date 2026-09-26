# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:51:29 2026

@author:  Itzu Huang
Topic: xml
"""
#import the xml package
import xml.etree.ElementTree as ET
#use the parse the load the data
tree=ET.parse("read.xml")

#find the root of the xml tree

root=tree.getroot()

#give a space list for the data
data=[]
#use the trace loop method to get each row of the root
#root here means the tag <dara> </data>
for row in root:
    sno=row.find("sno").text
    sna=row.find("sna").text
    tot=row.find("tot").text
    sarea=row.find("sarea").text
    ar=row.find("ar").text
    #after get the value of each row
    #use the append to put it into data
    data.append([sno,sna,tot,sarea,ar])

import pandas as pd
ubike=pd.DataFrame(data)
ubike.columns=["siteNo","siteName","total number","siteArea","Address"]
