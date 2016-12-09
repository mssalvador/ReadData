
'''
Created on Sep 7, 2016

@author: svanhmic
'''
import numpy as np
import sys
import itertools
from string import lower
from gtk.keysyms import Select
reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import DataFrameWriter,Row
from pyspark.sql.types import StringType, ArrayType, StructField, StructType,\
    IntegerType,FloatType
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from re import findall, sub
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab


fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
modelPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/"
scalerPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/StandardScalaer"

sc = SparkContext("local[8]","logisticModel")
sqlContext = SQLContext(sc)



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

def doPivotOnCol(df,column):
    ''' does a pivot operation on a particular column
        
        Note:
            df must have an index column as well, makes sure that it can be joined back to the mothership
        Args:
            df: dataframe containing atleast index and column, where column is the column that gets pivoted.
        
        Output: 
            pivDf a dataframe that contains number of distinct values from column in df now as rows where each element where element is in row is 1
    
    '''      
    pivDf = df.groupby("Index").pivot(column).count()
    for p in pivDf.columns:
        pivDf = pivDf.withColumnRenamed(p,column+":"+p)
    pivDf = pivDf.withColumnRenamed(column+":Index","Index")
    return pivDf

def createPivotDataset(df):
    ''' Creates Pivoted data set. Calls doPivotOnCol for each column that needs to be pivoted.
        
    Note:
        
    Args:
        df: dataframe with columns, those columns with type string or unicode gets pivoted
        
    Output: 
        data frame: with pivoted rows so should be a data frame with distinct(cols) + numeric cols number of columns.
    ''' 
    ColsDf = df.dtypes
    Cols = []
    for i in ColsDf:
        if i[1] == 'string':
            Cols.append(i[0])
        elif i[1] == 'unicode':
            Cols.append(i[0])

    for c in Cols:
        pivotData = doPivotOnCol(df,c).fillna(0)
        df = df.drop(c)
        df = df.join(pivotData, df.Index == pivotData.Index,'inner').drop(pivotData.Index)
    return df

def ImputerMean(df,cols):
    ''' Computes the mean for each columns and inserts the mean into each empty value in the column, if there are any.
        
    Note:
        
    Args:
        df: containing columns with numerical values.
        
    Output: 
        data frame with any empty values in columns filled out with mean
    ''' 
    for c in cols:
        describe = df.select(F.mean(df[c]).alias("mean")).collect()[0]["mean"]
        df = df.fillna(describe,c)
    return df

def toDense(col):
    
    return Vectors.dense(col.toArray())

udfToDense = F.udf(toDense, VectorUDT())

#Import the data!
virkData = sqlContext.read.format("json").load(fileStr+"/virksomhedersMetadata.json").cache() # loads the subsample of virksomheder  alleVirksomheder
#virkData.show(1)
#print virkData.columns

filteredVirkData = (virkData.filter( (virkData["sammensatStatus"] == "aktiv") | (virkData["sammensatStatus"] == "opl\xc3\xb8stefterkonkurs")  )
                    .drop(virkData["virksomhedsBeskrivelse"])
                    .drop(virkData["virksomhedsstatus"]))
#filteredVirkData.show(20,truncate=False)
#print filteredVirkData.count()

#some intital cleaning
minYear = virkData.groupBy().min("stiftelsesAar").collect()[0]
#print minYear

labeledDF = (filteredVirkData.select(filteredVirkData["*"],createLabelUdf(filteredVirkData["sammensatStatus"]).alias("label"))
            .drop(filteredVirkData["sammensatStatus"])
            )
#print labeledDF.dtypes

pivotFeaturesTemp = createPivotDataset(labeledDF)
columns = ["nPenheder","antalAnsatte","stiftelsesAar"]
pivotFeatures = ImputerMean(pivotFeaturesTemp,columns)
pivotCols = pivotFeatures.columns

labeledDFAlteredY = (pivotFeatures
                     .select(pivotFeatures["*"],(pivotFeatures["stiftelsesAar"]-F.lit(2000)).alias("aar"))
                     .drop(pivotFeatures["stiftelsesAar"])
                     .na.drop(how='any')
                     #.drop(pivotFeatures["brancheAnsvarskode:96"])
                     .drop(pivotFeatures["brancheAnsvarskode:97"])
                     .drop(pivotFeatures["brancheAnsvarskode:99"])).cache()

labeledDFAlteredY.describe().show() # a description of all data

#labeledDFAlteredY.show(20,truncate=False)
#labeledDFAlteredY.printSchema()
labelCol =  labeledDFAlteredY.columns
del labelCol[0] #remove index
del labelCol[1] #remove cvr
del labelCol[3] #remove label
#print labelCol

vecAssembler = VectorAssembler(inputCols=labelCol,outputCol="features")
trainData = vecAssembler.transform(labeledDFAlteredY)
denseTrainData = trainData.select(trainData["Index"],trainData["label"].cast("long").alias("label"),trainData["Features"])
#denseTrainData.show(20,truncate=False)
#denseTrainData.printSchema()

#lr = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="predictions", maxIter=10, regParam=0.3, elasticNetParam=0.8,fitIntercept=False)
glm = GeneralizedLinearRegression(labelCol="label", featuresCol="Features", family="binomial",link="logit")
#pipeline = Pipeline(stages=[vecAssembler,glm])
model = glm.fit(denseTrainData)
 
# Summarize the model over the training set and print out some metrics
summary = model.summary
print '%24s %20s %20s %20s %20s'  % (" ","Coefficients","Standard Error","T-values","P-values")
print '%24s %20s %20s %20s %20s'  % ("Intercept", str(model.intercept),str(summary.coefficientStandardErrors[-1]),str(summary.tValues[-1]),str(summary.pValues[-1]))
for i in range(0,len(labelCol)):
    print '%24s %20s %20s %20s %20s'  % (labelCol[i],str(model.coefficients[i]),str(summary.coefficientStandardErrors[i]),str(summary.tValues[i]),str(summary.pValues[i]))
print '%24s %20s' % ("Dispersion: ", str(summary.dispersion))
#print("Null Deviance: " + str(summary.nullDeviance))
print '%24s %20s' % ("Residual D.O.F Null:", str(summary.residualDegreeOfFreedomNull))
#print("Deviance: " ,summary.deviance)
print '%24s %20s' % ("Residual D.O.F:", str(summary.residualDegreeOfFreedom))
#print "AIC: " + str(summary.aic) 
print("Deviance Residuals: ")
summary.residuals().show()