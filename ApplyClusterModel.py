'''
Created on Aug 30, 2016

@author: svanhmic
'''

import numpy as np
import sys
import itertools
from string import lower
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, ArrayType
from array import ArrayType
reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import Row
import pyspark.sql.functions as F
from pyspark.ml.linalg import SparseVector,Vectors,VectorUDT
from pyspark.ml.feature import VectorAssembler, StandardScalerModel,StandardScaler
from pyspark.ml.clustering import KMeansModel, KMeans

import scipy.sparse as sps


fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
modelPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/"
scalerPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/StandardScalaer"
virksomheder = "/c1p_virksomhed.json"
alleVirksomheder = "/AlleDatavirksomheder.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"

sc = SparkContext("local[8]","Applyclustermodel")
sqlContext = SQLContext(sc)


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

def createSparseVector(array):
    ''' Takes a vector or list of vectors numpy.array and converts them into sparse vectors
        
    Note: This is an np.matrix
    Args:
        
        
    Output:     
    ''' 
    
    output = []
    for idx,row in enumerate(array):
        dic = {}
        length = len(row)
        for j,elem in enumerate(row):
            if elem > 0.0:
                dic[j] = elem
        output.append(Row(cluster=idx,coordinates=SparseVector(length,dic)))
    return output

def createDenseVector(SparseVectorDf):
    
    size = SparseVectorDf.size
    output = np.zeros(size)
    elements = zip(SparseVectorDf.indices,SparseVectorDf.values)
    
    for i,elm in enumerate(output):
        for j in elements:
            if i == j[0]:
                output[i] = j[1]
    return output

    

def absSubtract(v1, v2):
    """Add two sparse vectors
    >>> v1 = Vectors.sparse(3, {0: 1.0, 2: 1.0})
    >>> v2 = Vectors.sparse(3, {1: 1.0})
    >>> add(v1, v2)
    SparseVector(3, {0: 1.0, 1: 1.0, 2: 1.0})
    """
    #assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector), "v1s type: " +str(type(v1)) +" v2s type: " + str(type(v2))
    assert v1.size == v2.size  , "v1s size: " + str(v1.size) + " v2s size: " + str(v2.size)
    # Compute union of indices
    indices = set(v1.indices).union(set(v2.indices))
    # Not particularly efficient but we are limited by SPARK-10973
    # Create index: value dicts
    v1d = dict(zip(v1.indices, v1.values))
    v2d = dict(zip(v2.indices, v2.values))
    zero = np.float64(0)
    # Create dictionary index: (v1[index] + v2[index])
    values =  {i: np.abs(v1d.get(i, zero) - v2d.get(i, zero))
       for i in indices
       if np.abs(v1d.get(i, zero) - v2d.get(i, zero)) != zero}

    return Vectors.sparse(v1.size, values)

absSubUDF = F.udf(absSubtract,VectorUDT())

def computeDistance(col):
    assert isinstance(col, SparseVector), "Wrong! " + str(type(col))
    return float(col.norm(2))

computeEucleadianDist = F.udf(computeDistance,DoubleType())

def convertToSparse(size,indices,vals):
    assert len(indices) == len(vals) , "The lengths mismatch: " + str(len(indices)) + ", " + str(len(vals))
    return SparseVector(size,indices,vals)

convertUdfToSparse = F.udf(convertToSparse,VectorUDT())

def toA(col):
    return col.toArray()

def sparseSum(rdd):
    return np.sum(rdd) 


pivotFeatures = sqlContext.read.format("json").load(fileStr+"/pivotMetaData.json") # loads the subsample of virksomheder  alleVirksomheder
lScaledFeatures = sqlContext.read.format("json").load(fileStr+"/ScaldMetaData.json")
ScaledF = lScaledFeatures.select(lScaledFeatures["Index"],convertUdfToSparse(lScaledFeatures["scaledfeatures"]["size"]
                                                                             ,lScaledFeatures["scaledfeatures"]["indices"]
                                                                             ,lScaledFeatures["scaledfeatures"]["values"]).alias("scaledfeatures"))
#ModelScaler = StandardScaler(withMean=False,inputCol="features",outputCol="scaledfeatures").fit(ScaledF)
#ScaledF = ModelScaler.transform(ScaledF)
#print ScaledF.take(2)

loadKmeansModel = KMeansModel.read().load(modelPath+"Kmeans_Cvr_Model")
loadKmeans = KMeans.read().load(modelPath+"Kmeans_Cvr")
prediction = loadKmeansModel.transform(ScaledF)
#newModel = KMeans(featuresCol="scaledfeatures", predictionCol="predict", k=2, initMode="random")
#clusters = newModel.fit(ScaledF)
wssse = loadKmeansModel.computeCost(ScaledF) #computes the within sum of squares error
print("Within Set Sum of Squared Errors for k: "+ str(loadKmeans.getK()) +"  = " + str(wssse))
#print prediction.show(1,truncate=False)    
 
clusterCenters = loadKmeansModel.clusterCenters()
sparseClusterDf = sqlContext.createDataFrame(createSparseVector(clusterCenters))
#sparseClusterDf.show(1,truncate=False)
 
appendCentersDf = (prediction.select(prediction["Index"],prediction["scaledfeatures"],prediction["prediction"])
                   .join(sparseClusterDf,prediction["prediction"] == sparseClusterDf["cluster"],"inner")
                    )
#print appendCentersDf.dtypes
featureDistanceRDD = appendCentersDf.rdd.map(lambda x: Row(Index=x["Index"],FeatureContribution=absSubtract(x["scaledfeatures"],x["coordinates"]),Prediction=x["prediction"]))#.toDF(["Index","FeatureContribution","prediction"])
distDf = featureDistanceRDD.map(lambda x: Row(Index=x["Index"],FeatureContribution=x["FeatureContribution"],Prediction=x["Prediction"],Distance=computeDistance(x["FeatureContribution"]))).toDF()
#distDf.show(1)
featContri = distDf.rdd.map(lambda x: Row(p=x["Prediction"],f=Vectors.dense(toA(x["FeatureContribution"])))).toDF() # creates dense vector
featContri.show(1)
#predictAndDist = prediction.join(predictionDist,prediction.Index == predictionDist.Index,'inner').drop(predictionDist.Index)
#predictAndDist.show(1)
#orderedPredictedDist = predictAndDist.orderBy(predictAndDist["distance2center"],ascending=False)

#create the figure with subplots over distances to centers for each cluster.
# fig = plt.figure()
# c = 1
# fileSt = open(fileStr+"/results/test.csv","a+")
# for i in range(kmeans.getK()):
#     filteredPreds = predictAndDist.filter(predictAndDist["prediction"] == i).cache()
#     distances = filteredPreds.select(predictAndDist["distance2center"]).collect()
#     orderDists = (filteredPreds
#                   .orderBy(filteredPreds["distance2center"],ascending=False)
#                   .select(filteredPreds["cvrnummer"],log(filteredPreds["distance2center"]).alias("distance2center"))).take(20)
#     fileSt.write("cvr;afstand;cluster;"+str(i)+"\n")
#     for j in orderDists:
#         fileSt.write(str(j["cvrnummer"])+";"+str(j["distance2center"])+"\n")
#     fig = plt.subplot(3,3,c)
#     n, bins, patches = plt.hist(np.log(distances), 50, facecolor='green', alpha=0.75)
#     plt.ylabel('log antal')
#     plt.yscale('log', nonposy='clip')
#     plt.xlabel('log afstande til centrum')
#     plt.title(r'Afstand fra center til cluster-punkter cluster: %s '%(str(i)))
#     c += 1
# fileSt.close()
# plt.show()
#fig.savefig(fileStr+"/results/hist.png")
#orderedPredictedDist.drop(orderedPredictedDist["scaledfeatures"]).show(truncate=False)
#orderedPredictedDist.write.csv(fileStr+"results",mode="overwrite")   