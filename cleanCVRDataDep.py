'''
Created on Jun 21, 2016

@author: svanhmic
'''
from pyspark import SQLContext
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab
from pyspark import SparkContext
from pyspark.sql.functions import udf, log
from pyspark.sql import Row
from pyspark.ml.feature import HashingTF
from datetime import datetime
from pyspark.sql.types import StringType, IntegerType, ArrayType, FloatType, DateType

# HACK!!!
import sys
from string import lower
reload(sys)
sys.setdefaultencoding('utf-8')

#Spark startup stuff
sc = SparkContext("local[8]","virksomhedsTest")
sqlContext = SQLContext(sc)

#input stuff
virksomheder = "/c1p_virksomhed.json"
allVirksomheder = "/virksomhedsstatusAlle.json"
fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"


def closestWord(WordArr):
    #This method finds the word closest to 
    #
    #
    numFeatures = 22
    return [hash(x) for x in WordArr]

#longer functions

def deltaTimeString(row):
    fra = row["gyldigFra"]
    til = row["gyldigTil"]
    status = row["status"]
    delta = []
    for i in range(len(fra)):
        if fra[i] == None:
            fra[i] = "2100-01-01"
        if til[i] == None:
            til[i] = "2100-01-01"
        datediff = abs((datetime.strptime(til[i],"%Y-%m-%d")-datetime.strptime(fra[i],"%Y-%m-%d")).days)
        if datediff == 0:
            datediff = 1
        delta.append((datediff,status[i],row["cvr"]))
    return delta

def rowToList(rows):
    output = []
    for i in rows:
        output.append(i[0])
    return output

def getOptimalSubplots(leng):
    h = 0
    w = leng
    for x in range(1,leng):
        if w % x == 0:
            h = w / x
        if x >= h:
            w = x
            break
    print "width: ", w , " height: " , h
    return (h,w)

#udfs 
hashColumn = udf(closestWord,ArrayType(IntegerType(),False))
toLowerCase = udf(lambda x: [lower(i) for i in x],ArrayType(StringType(),False))

#Read the virksomheder file.
dataVirk = sqlContext.read.format("json").load(fileStr+allVirksomheder) # loads the subsample of virksomheder
print dataVirk.count()

#selecting and cleaning virksomheder to only include cvr, from and to dates and status
statusVirk= (dataVirk.select(
                dataVirk.cvr.alias("cvr")
                ,dataVirk.gyldigFra.alias("gyldigFra")
                ,dataVirk.gyldigTil.alias("gyldigTil")
                ,toLowerCase(dataVirk.status).alias("status"))
              )

#the old way of doing it
#statusVirk= (dataVirk.select(
#                dataVirk._source.Vrvirksomhed.cvrNummer.alias("cvr")
#                ,dataVirk._source.Vrvirksomhed.virksomhedsstatus.periode.gyldigFra.alias("gyldigFra")
#                ,dataVirk._source.Vrvirksomhed.virksomhedsstatus.periode.gyldigTil.alias("gyldigTil")
#                ,dataVirk._source.Vrvirksomhed.virksomhedsstatus.status.alias("status"))
#              )

mappedDiff = statusVirk.flatMap(lambda x: deltaTimeString(x)).collect()
statusDiff = sqlContext.createDataFrame(mappedDiff,["difference","status","cvr"])

#Group by status to get avg and count
avgStatusDiff = statusDiff.groupby(statusDiff["status"]).avg("difference")
countStatusDiff = statusDiff.groupby(statusDiff["status"]).count().alias("count")
distinctStatus = statusDiff.select(statusDiff["status"]).distinct().collect()

height, width = getOptimalSubplots(len(distinctStatus))
c = 1.0
fig = plt.figure()
for x in distinctStatus:
    
    normalDiff = statusDiff.filter(x[0] == statusDiff["status"]).select(log(statusDiff["difference"]).alias("difference")).cache()
    normalDescribe = normalDiff.describe().collect()
    collectedNormDiff = np.array([i[0] for i in normalDiff.collect()])

    #print x[0]
    if float(normalDescribe[1][1]) == 0:
        continue
    #normalDiff.describe().show()
    #print "w ",type(width) , " l " , type(height), " c " , type(c)
    fig = plt.subplot(width,height,c)
    n, bins, patches = plt.hist(collectedNormDiff, 50, normed=1, facecolor='green', alpha=0.75)

    y = mlab.normpdf( bins, float(normalDescribe[1][1]), float(normalDescribe[2][1]))
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.xlabel('Durration')
    plt.ylabel('Percent')
    plt.title(r'$\mathrm{Hist:\ %s \  Durration:}\ \mu=%s,\ \sigma=%s$'%(str(x[0]),normalDescribe[1][1][:4],normalDescribe[2][1][:4]))
    plt.axis([0, 15, 0, 2])
    plt.grid(True)#
    c = c + 1
    
plt.show()

