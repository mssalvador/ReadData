'''
Created on Feb 7, 2017

@author: svanhmic
'''

#scFeatRDD = scaledFeaturesDf.select(F.col("label"),F.col("features"))
#here comes the F-test
def computeFstat(df,cols=["prediction","features"],groupCols=["prediction"]):
    '''
        This method computes the one way F-statistic from:
        https://en.wikipedia.org/wiki/F-test

        Input: 
            df - Spark dataframe should at least contain cluster labels and a feature vector
            cols - list that contains the columns in the data frame to extract
            groupCols - list that contains the column (labels) that should be grouped when counting 
            the number of elements pr cluster

        Output:
            ratio - the F-statistic ratio
    ''' 
    countMap = (df
                .groupBy(*groupCols)
                .count()
                .rdd
                .map(lambda x: (x["prediction"],x["count"]))
                .collectAsMap()
               )
    broadCounts = sc.broadcast(countMap)

    partialMeansDf = (df  
                    .select([F.col(i) for i in cols])
                    .rdd
                    .map(lambda x: (x["prediction"],x["features"]))
                    .reduceByKey(lambda x,y:x+y)
                    .map(lambda x: (x[0],x[1]/broadCounts.value[x[0]]))
                    .toDF(["cluster","clusterMeans"])
                   )
    #print(partialMeansDf.take(10))
    totalMean = (df
                 .rdd
                 .map(lambda x: x["features"])
                 .reduce(lambda x,y:x+y)
                )
    #print(totalMean)
    totalLength = np.sum(list(broadCounts.value.values()))
    broadMean = sc.broadcast(totalMean/totalLength)

    groupWithInVariance = (partialMeansDf
                           .rdd
                           .map(lambda x: (x["cluster"], np.dot(x["clusterMeans"]-broadMean.value,x["clusterMeans"]-broadMean.value)))
                           .map(lambda x: broadCounts.value[x[0]]*x[1]/(len(broadCounts.value)-1) )
                           .reduce(lambda x,y:x+y)

                             )
    #print(groupWithInVarianceRdd.take(10))              

    squaredVectorSubtractUDF = F.udf(lambda x,y: float(np.dot(x-y,x-y)/(totalLength-len(broadCounts.value))),DoubleType())

    groupWithOutVariance = (df
                            .select([F.col(i) for i in cols])
                            .join(other=partialMeansDf,on=(partialMeansDf["cluster"]==df["prediction"]),how="left")
                            .drop(partialMeansDf["cluster"])
                            .withColumn(colName="newFeature",col=squaredVectorSubtractUDF(F.col("features"),F.col("clusterMeans")))
                            .groupBy()
                            .sum("newFeature")
                            .collect()
                           )
    #print(broadMean.value)
    return groupWithInVariance/groupWithOutVariance[0]["sum(newFeature)"]
#groupWithOutVariance.show()

if __name__ == '__main__':
    pass