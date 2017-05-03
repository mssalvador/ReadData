'''
Created on Jun 20, 2016

@author: svanhmic
'''

from pyspark import SparkContext
from pyspark.sql import SQLContext,Row,DataFrameWriter,DataFrameReader
from pyspark.rdd import RDD
import sys

#Applicable in Spark 2.0 >=
#spark = (SparkSession
#             .builder
#             .appName("regnData")
#             .getOrCreate())
sc = SparkContext(appName="regnData")
sqlContext = SQLContext(sc)
fileStr = ""
jsonFile = "cdata-permanent.json"
if len(sys.argv) == 1:
    fileStr = "/home/svanhmic/workspace/data/DABAI/sparkdata/json/"
else:
    fileStr = sys.argv[1]    


dataF = sqlContext.read.json(fileStr+jsonFile)
dataF.select(dataF["_type"]).distinct().show()
#print(dataF.columns)
#dataF.printSchema()
virksomhedsData =  dataF.filter(dataF._type == 'virksomhed').select(dataF['_source']["Vrvirksomhed"].alias("virksomhed"))
deltagerData = dataF.filter(dataF._type == 'deltager').select(dataF['_source']["Vrdeltagerperson"].alias("deltager"))
prodData = dataF.filter(dataF._type == 'produktionsenhed').select(dataF['_source']["VrproduktionsEnhed"].alias("prodenhed"))

#virksomhedsData.printSchema()
#print(dataF.count())
#types = dataF.select(dataF._type).distinct().collect()

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
deltagerData.write.json(fileStr+"/AlleDeltager.json",mode="overwrite")
prodData.write.json(fileStr+"/AlleProduktionEnheder.json",mode="overwrite")
#virksomhedsData.write.parquet(fileStr+"/AlleDatavirksomheder",mode="overwrite")