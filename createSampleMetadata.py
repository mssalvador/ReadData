'''
Created on Aug 24, 2016

@author: svanhmic
'''
import numpy as np
import sys

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.functions import  udf, mean
from re import findall, sub


fileStr = ""
virksomheder = "/c1p_virksomhed.json"
alleVirksomheder = "/AlleDatavirksomheder.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"

sc = SparkContext(appName="clustermodel")
sqlContext = SQLContext(sc)


def extractNumbers(col):
    
    try:
        if not isinstance(col, str): 
            ucol = str(col)
        else:
            ucol = col
        nums = findall('\d+',ucol)
        return int(nums[0])
    except:
        return np.nan
    
extractNumberUDF = udf(extractNumbers,IntegerType())

def getStrings(col):
    try:
        return str(col)
    except:
        return "None"

getStrings = udf(getStrings,StringType())

def extractStatus(col):
    return sub("\s+",'',col.lower())

extractStatusUDF = udf(extractStatus,StringType())

def extractVirkStatus(col):
    if len(col) == 0:
        return ""
    else:
        lastStatus = col[-1]["status"]
        return extractStatus(col[-1]["status"])

extractVirkStatusUDF = udf(extractVirkStatus, StringType())

def extractYearfromVirk(col):
    ''' Extracts years from date column in cvr
    
    Note:
        date format must be: yyyy-mm-dd or yyyy-dd-mm
    Args:
        Col: a column containing the date
    output:
        year: the year
    '''
    
    try:
        year = int(col[:4])
    except:
        year = np.nan
    return year

extractYearfromVirkUDF = udf(extractYearfromVirk,IntegerType())

def getIndex(rdd):
    ''' Extracts an index if it is in two colunmns e.g. [col1[col1.1,col1.2],col2]
        
        Args:
            rdd is elements in an rdd
        Output:
            r is an element in a rdd with is the two col1 and col2 appended into [col1.1, col1.2 , col2, ...]
    '''
    r = [rdd[1]]
    for i in rdd[0]:
        r.append(i)
    return r

if __name__ == '__main__':
    
    if len(sys.argv) == 0:
        fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
    else:
        fileStr = sys.argv[1]
    
    virkData = sqlContext.read.format("json").load(fileStr+alleVirksomheder) # loads the subsample of virksomheder  alleVirksomheder
    virkDataTemp = virkData.select(virkData["virksomhed"]["virksomhedMetadata"].alias("metadata")
                                   ,virkData["virksomhed"]["virksomhedsstatus"].alias("virksomhedsstatus")
                                   ,virkData["virksomhed"]["brancheAnsvarskode"].alias("brancheAnsvarskode")
                                   ,virkData["virksomhed"]["reklamebeskyttet"].alias("reklamebeskyttet")
                                   ,virkData["virksomhed"]["cvrnummer"].alias("cvrnummer"))
    
    
    virkDataDf = virkDataTemp.select( virkDataTemp["metadata"]["antalPenheder"].alias("nPenheder")
                                     ,virkDataTemp["metadata"]["nyesteAarsbeskaeftigelse"]["intervalKodeAntalAnsatte"].alias("aarsbeskfansat")
                                     #,virkDataTemp["metadata"]["nyesteBibranche1"]["branchekode"].alias("branchekode1")   
                                     #,virkDataTemp["metadata"]["nyesteBibranche1"]["branchetekst"].alias("branchetekst1")
                                     #,virkDataTemp["metadata"]["nyesteBibranche2"]["branchekode"].alias("branchekode2")
                                     #,virkDataTemp["metadata"]["nyesteBibranche2"]["branchetekst"].alias("branchetekst2")
                                     #,virkDataTemp["metadata"]["nyesteBibranche3"]["branchekode"].alias("branchekode3")
                                     #,virkDataTemp["metadata"]["nyesteBibranche3"]["branchetekst"].alias("branchetekst3")                                     
                                     #,virkDataTemp["metadata"]["nyesteHovedbranche"]["branchekode"].alias("hovedbranchekode")
                                     #,virkDataTemp["metadata"]["nyesteStatus"]["statuskode"].alias("statuskode")
                                     ,virkDataTemp["metadata"]["nyesteVirksomhedsform"]["langBeskrivelse"].alias("langBeskrivelse")
                                     ,virkDataTemp["metadata"]["nyesteVirksomhedsform"]["virksomhedsformkode"].alias("virksomhedsformkode")
                                     ,virkDataTemp["metadata"]["sammensatStatus"].alias("sammensatStatus")
                                     ,virkDataTemp["metadata"]["stiftelsesDato"].alias("stiftelsesDato")
                                     ,virkDataTemp["cvrnummer"]
                                     ,virkDataTemp["virksomhedsstatus"]
                                     ,virkDataTemp["brancheAnsvarskode"]
                                     ,virkDataTemp["reklamebeskyttet"].cast('integer').alias("reklamebeskyttet")
                                     )
    #print virkDataDf.dtypes
    #virkDataDf.show(truncate=False)
    
    IndexedVirkDf = (virkDataDf.select(virkDataDf["cvrnummer"]
                                       ,virkDataDf["nPenheder"]
                                       ,extractNumberUDF(virkDataDf["aarsbeskfansat"] ).alias("antalAnsatte")
                                       #,getStrings(virkDataDf["branchekode1"]).alias("branchekode1")
                                       #,getStrings(virkDataDf["branchekode2"]).alias("branchekode2")
                                       #,getStrings(virkDataDf["branchekode3"]).alias("branchekode3")
                                       #,getStrings(virkDataDf["hovedbranchekode"]).alias("hovedbranchekode")
                                       #,getStrings(virkDataDf["statuskode"]).alias("statuskode")
                                       ,getStrings(virkDataDf["langBeskrivelse"]).alias("virksomhedsBeskrivelse")
                                       ,getStrings(virkDataDf["brancheAnsvarskode"]).alias("brancheAnsvarskode")
                                       ,extractStatusUDF(virkDataDf["sammensatStatus"]).alias("sammensatStatus")
                                       #,virkDataDf["virksomhedsformkode"]
                                       #,virkDataDf["stiftelsesDato"]
                                       ,extractYearfromVirkUDF(virkDataDf["stiftelsesDato"]).alias("stiftelsesAar")                                   
                                       ,extractVirkStatusUDF(virkDataDf["virksomhedsstatus"]).alias("virksomhedsstatus")
                                       ,virkDataDf["reklamebeskyttet"]
                                       )).rdd.zipWithIndex().map(lambda x: getIndex(x))
                                       
    cleanedVirkDf = sqlContext.createDataFrame(IndexedVirkDf,["Index","cvrnummer","nPenheder","antalAnsatte"#,"branchekode1","branchekode2","branchekode3","hovedbranchekode"
                                                              #,"statuskode"
                                                              ,"virksomhedsBeskrivelse"
                                                              ,"brancheAnsvarskode","sammensatStatus"
                                                              #,"virksomhedsformkode"
                                                              ,"stiftelsesAar","virksomhedsstatus","reklamebeskyttet"])
    
    #virkData.printSchema()
    
    #data is pivoted doing the "bag of words" and afterwards doing an imputer mean which substitutes a mean value for a column in an null value element of that col.
    
    cleanedVirkDf.write.json(fileStr+"/virksomhedersMetadata.json",mode="overwrite")
