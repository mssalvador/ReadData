'''
Created on Aug 19, 2016

@author: svanhmic
'''

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import mean, lit
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler


fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
modelPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/"
scalerPath = "/home/svanhmic/workspace/Python/Erhvervs/Models/StandardScalaer"

sc = SparkContext("local[8]","clustermodel")
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
        describe = df.select(mean(df[c]).alias("mean")).collect()[0]["mean"]
        df = df.fillna(describe,c)
    return df

virkData = sqlContext.read.format("json").load(fileStr+"/virksomhedersMetadata.json") # loads the subsample of virksomheder  alleVirksomheder
#virkData.printSchema()

#data is pivoted doing the "bag of words" and afterwards doing an imputer mean which substitutes a mean value for a column in an null value element of that col.
pivotFeaturesTemp = createPivotDataset(virkData)
columns = ["nPenheder","antalAnsatte","stiftelsesAar"]
pivotFeatures = ImputerMean(pivotFeaturesTemp,columns)
pivotCols = pivotFeatures.columns
pivotCols.remove("Index")
pivotCols.remove("cvrnummer")
pivotFeatures.write.json(fileStr+"/pivotMetaData.json",mode="overwrite") # write the raw feature data to json file
#print pivotCols
#pivotFeatures.show(1,truncate=False)

vectorizer = VectorAssembler()
vectorizer.setInputCols(pivotCols) # sets the input cols names as input to the method
vectorizer.setOutputCol('features') # names the output col: Features
tempVectorOutput = vectorizer.transform(pivotFeatures) # converts the dataframe of cols to a dataframe with one col containing vector
vectorOutput = tempVectorOutput.select(tempVectorOutput["Index"],tempVectorOutput["features"])


scaledModel = StandardScaler(withMean=False,inputCol=vectorizer.getOutputCol(),outputCol="scaledfeatures").fit(vectorOutput) # Generates a model for scaled data arround mean and with std
scaledOutput = scaledModel.transform(vectorOutput)
scaledFeatures = scaledOutput.drop(scaledOutput["features"])
scaledFeatures.show(1,truncate=False)
scaledFeatures.write.json(fileStr+"/ScaldMetaData.json",mode="overwrite")
scaledModel.write().overwrite().save(scalerPath)
#print scaledFeatures.show(1,truncate=False)

params = [4,8,10,12,16]
kmeans = KMeans(k=2, maxIter=10,featuresCol="scaledfeatures",initMode="random")
#bestFit = (0,10000000)
#for p in params:  
#    #print scaledFeatures.take(1)
#    clusters = kmeans.fit(scaledFeatures)
#    prediction = clusters.transform(scaledFeatures)
#    wssse = clusters.computeCost(scaledFeatures) #computes the within sum of squares error
#    print("Within Set Sum of Squared Errors for k: "+ str(kmeans.getK()) +"  = " + str(wssse))
#    if bestFit[1] > wssse:
#        bestFit = (kmeans.getK(),wssse)
#    kmeans.setK(p)

#print bestFit[0] , bestFit[1]
kmeans.setK(params[1])
clusters = kmeans.fit(scaledFeatures)
kmeans().overwrite().save(modelPath+"Kmeans_Cvr")
clusters.write().overwrite().save(modelPath+"Kmeans_Cvr_Model")
