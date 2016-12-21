'''
Created on Jun 20, 2016

@author: svanhmic
'''


from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions

fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
virksomheder = "/c1p_virksomhed.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"
sc = SparkContext("local[8]","partionedCvr")
sqlContext = SQLContext(sc)

dataProd = sqlContext.read.format("json").load(fileStr+produktionsenhed)
dataVirk = sqlContext.read.format("json").load(fileStr+virksomheder)
dataDelt = sqlContext.read.format("json").load(fileStr+deltager)
#Fool arround with the data; find out what's what...
#print type(dataF) #Dataframe
#print dataF.count()
#print dataF.first()
#print type(listedData)
#print dataF.columns
cvrNumbersProd = dataProd.select('_source').persist()
cvrNumbersVirk = dataVirk.select('_source').persist()
cvrNumbersDelt = dataDelt.select('_source').persist() 
print(type(cvrNumbersVirk.select('_source.Vrvirksomhed')))
print(cvrNumbersVirk.select('_source.Vrvirksomhed.virksomhedsstatus','_source.Vrvirksomhed.aarsbeskaeftigelse').take(1))

#with open("/home/svanhmic/workspace/Python/ErhvervStyrrelsenDown/data/cvrdata/static/skemaDeltager.txt","w") as f:
#    f.write(str(cvrNumbersDelt.dtypes))