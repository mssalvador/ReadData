'''
Created on Aug 24, 2016

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
from pyspark.sql import DataFrameWriter
from pyspark.sql.types import StringType, ArrayType, StructField, StructType,\
    IntegerType,FloatType
from pyspark.sql.functions import explode, udf,coalesce,split,\
    mean,log
from pyspark.mllib.linalg import SparseVector, VectorUDT, Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from re import findall, sub


fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
virksomheder = "/c1p_virksomhed.json"
alleVirksomheder = "/AlleDatavirksomheder.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"

sc = SparkContext("local[8]","clustermodel")
sqlContext = SQLContext(sc)


def extractNumbers(col):
    
    try:
        if type(col) is not unicode: 
            ucol = unicode(col)
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
    return sub("\s+",'',lower(col))

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


virkData = sqlContext.read.format("json").load(fileStr+alleVirksomheder) # loads the subsample of virksomheder  alleVirksomheder
virkMetaData = (virkData.select(virkData["virksomhed"]["virksomhedMetadata"].alias("virkmetadata")
                                ,virkData["virksomhed"]["cvrNummer"].alias("cvrnummer")
                                ,virkData["virksomhed"]["virksomhedsstatus"].alias("virksomhedsstatus")
                                ,virkData["virksomhed"]["reklamebeskyttet"].alias("reklamebeskyttet")
                                ,virkData["virksomhed"]["brancheAnsvarskode"].alias("brancheAnsvarskode")
                                )) #exectracts metadata

virkDataDf = virkMetaData.select( virkMetaData["virkmetadata"]["antalPenheder"].alias("nPenheder")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["intervalKodeAntalAnsatte"].alias("aarsbeskfansat")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche1"]["branchekode"].alias("branchekode1")   
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche1"]["branchetekst"].alias("branchetekst1")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche2"]["branchekode"].alias("branchekode2")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche2"]["branchetekst"].alias("branchetekst2")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche3"]["branchekode"].alias("branchekode3")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche3"]["branchetekst"].alias("branchetekst3")                                     
                                 ,virkMetaData["virkmetadata"]["nyesteHovedbranche"]["branchekode"].alias("hovedbranchekode")
                                 ,virkMetaData["virkmetadata"]["nyesteStatus"]["statuskode"].alias("statuskode")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["langBeskrivelse"].alias("langBeskrivelse")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["virksomhedsformkode"].alias("virksomhedsformkode")
                                 ,virkMetaData["virkmetadata"]["sammensatStatus"].alias("sammensatStatus")
                                 ,virkMetaData["virkmetadata"]["stiftelsesDato"].alias("stiftelsesDato")
                                 ,virkMetaData["cvrnummer"]
                                 ,virkMetaData["virksomhedsstatus"]
                                 ,virkMetaData["brancheAnsvarskode"]
                                 ,virkMetaData["reklamebeskyttet"].cast('integer').alias("reklamebeskyttet")
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

cleanedVirkDf.write.json(fileStr+"/virksomhedersMetadata.json",mode="overwrite")