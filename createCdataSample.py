'''
Created on Jun 20, 2016

@author: svanhmic
'''

from pyspark import SparkContext
from pyspark.sql import SQLContext,Row,DataFrameWriter,DataFrameReader
from pyspark.rdd import RDD
import sys


<<<<<<< HEAD
sc = SparkContext(appName="regnData")
=======
sc = SparkContext("local[8]","regnData",app)
>>>>>>> e14f50ae3a930c5e7af2d3c1866ca2d6954f1910
sqlContext = SQLContext(sc)
fileStr = ""

if len(sys.argv) == 0:
    fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
else:
    fileStr = sys.argv[1]


dataF = sqlContext.read.json(fileStr+"/cdata-permanent.json")
virksomhedsData =  dataF.filter(dataF._type == 'virksomhed').select(dataF['_source']["Vrvirksomhed"].alias("virksomhed"))
virksomhedsData.printSchema()
print(dataF.count())
types = dataF.select(dataF._type).distinct().collect()

#virkNumb = dataF.filter(dataF._type == 'virksomhed').select('_source.Vrvirksomhed.cvrNummer').persist()
#print virkNumb.count()
#print virkNumb.distinct().count()
#print virkNumb.take(4)

#for r in types:
#    smallSet = dataF.filter(dataF._type == str(r["_type"])).sample(False,0.01,42) # collect small sample 
#    smallSet.write.json(fileStr+"/c1p_"+str(r["_type"])+".json",mode="overwrite")

#smallerset = (virksomhedsData
#              .select(virksomhedsData.virksomhed.cvrNummer.alias("cvr"),
#                      virksomhedsData.virksomhed.virksomhedsstatus.status.alias("status"),
#                      virksomhedsData.virksomhed.virksomhedsstatus.periode.gyldigFra.alias("gyldigFra"),
#                      virksomhedsData.virksomhed.virksomhedsstatus.periode.gyldigTil.alias("gyldigTil"))
#              )
#smallerset.write.json(fileStr+"/AlleDatavirksomheder.json",mode="overwrite")
virksomhedsData.write.json(fileStr+"/AlleDatavirksomheder.json",mode="overwrite")
