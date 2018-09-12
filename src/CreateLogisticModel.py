'''
Created on Sep 7, 2016

@author: svanhmic
'''
import numpy as np
import sys
import itertools
from string import lower
reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import DataFrameWriter,Row
from pyspark.sql.types import StringType, ArrayType, StructField, StructType,\
    IntegerType,FloatType
from pyspark.sql import functions as F
from pyspark.mllib.linalg import SparseVector, VectorUDT, Vectors
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from re import findall, sub
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab


fileStr = "/home/svanhmic/workspace/data/DABAI/sparkdata/json"
modelPath = "/home/svanhmic/workspacedata/DABAI/models/"
scalerPath = "/home/svanhmic/workspace/DABAI/models/StandardScalaer"

sc = SparkContext("local[8]","logisticModel")
sql_Context = SQLContext(sc)



def makeInteger(col):
    if col == "None":
        return 100
    else:
        return int(col)
makeUdfInt = F.udf(makeInteger, IntegerType())

def createLabel(col):
    if col == "aktiv":
        return 0
    else:
        return 1
     
createLabelUdf = F.udf(createLabel, IntegerType())

virkData = sql_Context.read.format("json").load(fileStr + "/virksomhedersMetadata.json").cache() # loads the subsample of virksomheder  alleVirksomheder
virkData.show(1)
#print virkData.columns
# some summery of whats in virk status
#virkStatus = sorted(virkData.groupBy(virkData["virksomhedsstatus"]).count().collect())
#for i in virkStatus:
#    print str(i["virksomhedsstatus"]) ," : " , str(i["count"])
#virkStatusKode = sorted(virkData.groupBy(virkData["virksomhedsstatus"]).count().collect())
#for i in virkStatus:
#    print str(i["virksomhedsstatus"]) ," : " , str(i["count"])
    

#pivotFeatures = sqlContext.read.format("json").load(fileStr+"/pivotMetaData.json") # loads the subsample of virksomheder  alleVirksomheder
#pivotFeatures.show(1)
#print [i.encode("utf-8") for i in pivotFeatures.columns]

filteredVirkData = virkData.filter( (virkData["sammensatStatus"] == "aktiv") | (virkData["sammensatStatus"] == "opl\xc3\xb8stefterkonkurs")  )
#filteredVirkData.show(20,truncate=False)
#print filteredVirkData.count()

minYear = virkData.groupBy().min("stiftelsesAar").collect()[0]
#print minYear

labeledDF = (filteredVirkData.select(filteredVirkData["*"],createLabelUdf(filteredVirkData["sammensatStatus"]).alias("label"))
             .drop(filteredVirkData["sammensatStatus"])
             .drop(filteredVirkData["virksomhedsBeskrivelse"])
             .drop(filteredVirkData["virksomhedsstatus"]))


labeledDFAlteredY = (labeledDF
                     .select(labeledDF["*"],(labeledDF["stiftelsesAar"]-F.lit(2000)).alias("aar"),makeUdfInt(labeledDF["brancheAnsvarskode"]).alias("ansvarskode"))
                     .drop(labeledDF["stiftelsesAar"])
                     .drop(labeledDF["brancheAnsvarskode"])
                     .na.drop())
#labeledDFAlteredY.show(20,truncate=False)

vecAssembler = VectorAssembler(inputCols=["antalAnsatte","ansvarskode","nPenheder","reklamebeskyttet","aar"],outputCol="features")
trainData = vecAssembler.transform(labeledDFAlteredY)
trainData.printSchema()
#lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="predictions", maxIter=10, regParam=0.3, elasticNetParam=0.8,fitIntercept=False)
glm = GeneralizedLinearRegression(labelCol="label", featuresCol="features", predictionCol="predictions", family="binomial",link="logit")
#pipeline = Pipeline(stages=[vecAssembler,glm])

model = glm.fit(trainData)
#model = pipeline.fit(labeledDFAlteredY)
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

# Summarize the model over the training set and print out some metrics
summary = model.summary

print "Coefficient Standard Errors: " + str(summary.coefficientStandardErrors)
print "T Values: " + str(summary.tValues)
print "P Values: " + str(summary.pValues)
print "Dispersion: " + str(summary.dispersion)
print("Null Deviance: " + str(summary.nullDeviance))
print "Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull)
#print("Deviance: " + str(summary.deviance))
print "Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom)
#print("AIC: " + str(summary.aic))
print("Deviance Residuals: ")
summary.residuals().show()
print "Deviance : " + summary.residuals().groupBy().sum("devianceResiduals")













