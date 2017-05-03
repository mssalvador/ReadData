'''
Created on Feb 7, 2017

@author: svanhmic
'''


from sklearn.linear_model import LogisticRegression as skLogistic
from pyspark.mllib.linalg import Vectors as oVector, VectorUDT as oVectorUDT
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
from pyspark import SparkContext


from spark_sklearn import GridSearchCV,Converter
from sklearn.cluster import KMeans as skKmeans
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import json
import sys



def getConfusion(label,prediction):
    diff = abs(label-prediction)
    if (diff == 0) and (label == 0):
        return "TN"
    elif (diff == 0) and (label == 1):
        return "TP"
    elif (diff == 1) and (label == 0):
        return "FP"
    elif (diff == 1) and (label == 1):
        return "FN"
    else:
        return "Excluded"

subUdf = F.udf(lambda x,y: getConfusion(x,y),StringType())


def computeConfusion(df):
    cols = [F.col(i) for i in ("cvrNummer","label","predictionLogReg")]
    return (df
            .select(*cols,subUdf(F.col("label"),F.col("predictionLogReg")).alias("diff"))
            .groupby().pivot("diff",["TP","TN","FN","FP","Excluded"]).count()
            .withColumn(col=((F.col("TP")+F.col("TN"))/(df.count()-F.col("Excluded"))),colName="accuracy")
           )


def getStatus(df):
    return computeConfusion(df)

def showStats(df):
    accuracyDf = computeConfusion(df)
    print(accuracyDf.head())
    accDf = (accuracyDf
             .select()
             .collect()[0][0])
    print("Accuracy: "+str(accDf))

def createPandasDf(sc,df,featuresCol="features",idCol="cvrNummer",*y,**x):
    '''
        improved convert to pandas dataframe
        index and features are the bare minimum
    '''
    
    #print(type(df))
    dfCols = df.columns
    columnsInX = [i for i in list(x.values()) if i in dfCols]
    columnsInY = [i for i in y if (i in dfCols) and (i not in columnsInX ) ]
    allColumns = columnsInX+columnsInY
    try:
        allColumns.remove(featuresCol)
        allColumns.remove(idCol)
    except ValueError:
        print("no extra cols")
        
    featDict = checkVectorTypes(df.schema,featuresCol)
    toOldVectorUdf = isVectorMLLib(featDict[1])
    conv = Converter(sc)
    return conv.toPandas(df.select([idCol,toOldVectorUdf(featDict[0]).alias(featuresCol)]+allColumns))  

def checkVectorTypes(schema,featureList="features"):

    assert schema.needConversion() == True , "it is not good"
    dd = json.loads(schema.json())
    mappedJson = map(lambda x: x["name"],dd["fields"])
    assert featureList in mappedJson, featureList+" is not there."
    
    try:
        return list(map(lambda x: (x["name"],x["type"]["pyClass"]),filter(lambda x: x["name"] == featureList,dd["fields"])))[0]
    except KeyError:
        print("hmm")
        return list(map(lambda x: (x["name"],x["type"]["class"]),filter(lambda x: x["name"] == featureList,dd["fields"])))[0]

def isVectorMLLib(typ):
    if typ in "pyspark.ml.linalg.VectorUDT":
        return F.udf(lambda x: oVector.dense(x.toArray()),oVectorUDT())
    else:
        return F.udf(lambda x: x,oVectorUDT())
    
def computeMaxSlope(slopes):
    '''
        Basically computes the simple slope of a series of points
        Input:
            slopes - npvector containing x,y vals
            
        Output:
            highest parameter pair slope
    '''
    diffSlops = np.diff(slopes[:,1])
    diffs = list(zip(slopes[1:,0],diffSlops))
    RowCol =  np.where(diffs == np.max(diffSlops))
    return diffs[RowCol[0][0]]

def labelOutliers(df,predictionCol="predictionKmeans",labelCol="label",threshold = 0.005,iterations=1):
    '''
        This method will label the data as outliers in the dataframe
        
        Input:
            df - pandas dataframe 
        Output:
            df_out - pandas dataframe with outliers as labels
    '''
    
    assert "isOutlier" not in df.column, "Error isOutlier is not "
    
    
    groupedClusters = df.groupby(predictionCol,as_index=False).count()
    outliers = groupedClusters[groupedClusters[labelCol]<=threshold*len(df)]
    #print(outliers["predictionKmeans"].values.tolist())
    df.ix[df["predictionKmeans"].isin(outliers["predictionKmeans"].values.tolist()),"isOutlier"] = iterations
    return df


def trainGridAndEval(sc,df,featureCol="features",parms = {'n_clusters':(2,3,4),}):
    '''
    This method should contain the kmeans gridsearch function  from spark_sklearn, 
    such that the gridsearch is paralized.
    
    Input: 
        df- Spark dataframe. Df should contain features in a ml dense vector format or mllib format
        parms - Dictionary with parameters for the machine learning algorithm
        **dfParams - Dictionary containing various information, e.g. columnsnames for feature and id
        more can be added
    
    output: returns the same data frame with predictions. 
    '''
    #args is used here!
    #panScaledDf = createPandasDf(df,featureCol,idCol,dfParams)
    
    #Initialize the gridsearch
    gs = GridSearchCV(sc,skKmeans(random_state=True),parms,error_score="numeric",cv=10)
    #print(type(x))
    #print(type(features))
    #model =
    return  gs.fit(df[featureCol].tolist()) #panScaledDf.assign(predict=model.predict(x))
  

def onePass(sc,df,params={'n_clusters':(2,3),},featureCol="features",idCol="cvrNummer",labelCol="label",*extraDf):
    '''
        The famous method
    '''
    
    #testing if df has isoutliers column
    assert ("isOutlier" in df.columns), "isOutlier is not in data frame: "+str(df.columns)
    
    #create holdout dataframe to previous iterations
    notOutliersDf = df[df["isOutlier"] == 0] 
    outliersDf = df[df["isOutlier"] > 0] 
    
    try:
        del notOutliersDf["predictionKmeans"]
        del notOutliersDf["predictionLogReg"]
    except:
        print(str(df.columns))

    
    #use trainGridAndEval to find the best parameters for the clustering method
    model = trainGridAndEval(sc,notOutliersDf,featureCol = featureCol,parms = params)
    
    #extract the "best" parameters using the elbow-method
    means = map(lambda x: [x[0]["n_clusters"],x[1]],model.grid_scores_)
    clusters = np.array(list(means))
    bestClusterParams = computeMaxSlope(clusters)
    
    #run the kmeans model again, yes this is stupid, but i cannot get the elbow best
    #parameter out of spark_sklearn, yet.
    
    length = len(notOutliersDf)
    bestKmeans = skKmeans(n_clusters = int(bestClusterParams[0]),random_state = True)
    bestpredictionKmeans = bestKmeans.fit_predict(notOutliersDf[featureCol].tolist())
    panScaledDf = notOutliersDf.assign(predictionKmeans = bestpredictionKmeans)
    
    #comence the cutoff
    groupedClusters = panScaledDf.groupby("predictionKmeans",as_index = False).count()
    outliers = groupedClusters[groupedClusters["label"] <= float(extraDf[1])*length]
    #print(outliers["predictionKmeans"].values.tolist())
    
    panScaledDf.ix[panScaledDf["predictionKmeans"].isin(outliers["predictionKmeans"].values.tolist()),"isOutlier"] = int(extraDf[0])
    panScaledDf = panScaledDf.assign(predictionLogReg = np.full(length,np.nan))
    #print(outliers.columns)
    
    #comence the logisticRegression train
    trainPDf = panScaledDf[panScaledDf["isOutlier"] == 0]
    notOutliersDfTrainCv,notOutliersDfTest = train_test_split(trainPDf,test_size = 0.2,stratify=trainPDf["label"])
    
    #concate the two hold out dataframes together
    outliersDf = outliersDf.append(panScaledDf[panScaledDf["isOutlier"] > 0],ignore_index = True) 
    
    print("Remaining data points: "+str(len(notOutliersDfTrainCv)))
    print("Total data points: "+str(len(panScaledDf)))
    print("Excluded data points: "+str(len(outliersDf)))
    
    logisticParams = {"C":(0.1,0.333,0.666,0.999),}
    logsiticGS = GridSearchCV(sc,skLogistic(),logisticParams)
    bestpredictionLogReg = logsiticGS.fit(notOutliersDfTrainCv[featureCol].tolist(),notOutliersDfTrainCv[labelCol])
    notOutliersDfTest = notOutliersDfTest.assign(predictionLogReg = bestpredictionLogReg.predict(notOutliersDfTest[featureCol].tolist()))
    
    return pd.concat([notOutliersDfTrainCv,notOutliersDfTest,outliersDf])

if __name__ == '__main__':
    sc = SparkContext(appName="regnData")
    sqlContext = SQLContext(sc)
    PATH = "/home/svanhmic/workspace/data/DABAI/sparkdata/json"
    
    
