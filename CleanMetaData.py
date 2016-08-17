'''
Created on Aug 16, 2016

This program extracts and cleans metadata

@author: svanhmic
'''

import sys
import string
from pyspark.ml.linalg import VectorUDT

reload(sys)
sys.setdefaultencoding('utf-8')

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import Row, Column
from pyspark.sql.types import StringType, ArrayType, StructField, StructType
from pyspark.sql.functions import explode, udf,coalesce
from pyspark.mllib.linalg import SparseVector, VectorUDT

fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
virksomheder = "/c1p_virksomhed.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"

sc = SparkContext("local[8]","partionedCvr")
sqlContext = SQLContext(sc)

def parseMetaData(col1,col2):
    if col1 is None  or col2 is None:
        bigRow = 0
    else:
        bigRow = []
        bigRow.append(col1)
        bigRow.append(col2)
    return bigRow

def columnsToList(row):
    #converts a flat dataframe with columns to a data frame with one column with lists as rows.
    #args df dataframe with data
    #output df with one column containing list with previous columns
    #colList = df.columns
    featlist=[x for x in row]
    for (i,f) in enumerate(featlist):
        if f == None:
            featlist[i] = None
        elif type(f) is not unicode:
            featlist[i] = unicode(f)
    return Row(featlist = featlist)

def bagOfWords(rawFeats, bagDict,lenOfDicts):
    """ Produces bag of words from a list of raw words and a dictionary
    
    Note:
        Indicies should always be sorted when using sparse vector!!!
        
    Args:
        rawFeats a list of unicode values. bagDict a dictionary of distinct words that rawFeats gets compared up against. 
        lenOfDicts the length of the dictonary, which is the length of the dictionary
    Returns:
        A SparseVector with length lenOfDicts, with indicies active if rawFeats is present and with value = count of types rawFeats
    
    """
    
    dicts = {}
    for x in rawFeats:
        if not dicts.has_key(bagDict.value[x]):
            dicts[bagDict.value[x]] = 1
        else:
            dicts[bagDict.value[x]] += 1   
        
    return SparseVector(lenOfDicts,dicts)
    
def bagOfWordsGenerator(broadcastDictOfWords):
    """ Generates UDF on bag of words from a list of raw words and a dictionary
    
    Note:
        
    Args:
        broadcastDictOfWords 
    Returns:
       UDF which can be used in dataframe select statement
    """
    
    leng = len(broadcastDictOfWords.value)
    return udf(lambda x: bagOfWords(x,broadcastDictOfWords,leng),VectorUDT())


columnsToListUDF = udf(columnsToList,StringType())
parseMetaDataUDF = udf(parseMetaData, ArrayType(StringType(),containsNull=True))
    

rawData = sqlContext.read.format("json").load(fileStr+onePercent) # loads the subsample of virksomheder
virkData = (rawData
            .where(rawData["_source"]["Vrvirksomhed"].isNotNull())
            .select(rawData["_source"]["Vrvirksomhed"].alias("virksomhed"))
            )
virkMetaData = (virkData.select(virkData["virksomhed"]["virksomhedMetadata"].alias("virkmetadata"))).cache() #exectracts metadata
virkNulldf = virkMetaData.select(virkMetaData["virkmetadata"].isNull().alias("isnull"),virkMetaData["virkmetadata"])
#virkNulldf.printSchema()

#normalize the entire metadataset
virkDataDf = virkMetaData.select( virkMetaData["virkmetadata"]["antalPenheder"].alias("nPenheder")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["aar"].alias("aarsbeskfaar")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["intervalKodeAntalAarsvaerk"].alias("aarsbeskfaarsvrk")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["intervalKodeAntalAnsatte"].alias("aarsbeskfansat")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["intervalKodeAntalInklusivEjere"].alias("aarsbeskfinkejer")
                                 ,virkMetaData["virkmetadata"]["nyesteAarsbeskaeftigelse"]["sidstOpdateret"].alias("aarsbeskefopdat")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["bynavn"].alias("by")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["etage"].alias("etage")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["husnummerFra"].alias("husnummerfra")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["husnummerFra"].alias("husnummertil")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["kommune"]["kommuneKode"].alias("kommunekode")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["kommune"]["kommuneNavn"].alias("kommunenavn")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["landekode"].alias("landkode")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["postboks"].alias("postboks")                                 
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["postnummer"].alias("postnummer")
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["vejkode"].alias("vejkode")                                                                    
                                 ,virkMetaData["virkmetadata"]["nyesteBeliggenhedsadresse"]["vejnavn"].alias("vejnavn")#branchetekst
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche1"]["branchekode"].alias("branchekode1")   
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche1"]["branchetekst"].alias("branchetekst1")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche2"]["branchekode"].alias("branchekode2")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche2"]["branchetekst"].alias("branchetekst2")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche3"]["branchekode"].alias("branchekode3")
                                 ,virkMetaData["virkmetadata"]["nyesteBibranche3"]["branchetekst"].alias("branchetekst3")                                     
                                 ,virkMetaData["virkmetadata"]["nyesteHovedbranche"]["branchekode"].alias("hovedbranchekode")
                                 ,virkMetaData["virkmetadata"]["nyesteHovedbranche"]["branchetekst"].alias("hovedbranchetekst")
                                 ,explode(virkMetaData["virkmetadata"]["nyesteKontaktoplysninger"]).alias("kontakt")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["aar"].alias("kvarbeskfaar")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["kvartal"].alias("kvarbeskfkvart")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["intervalKodeAntalAarsvaerk"].alias("kvarbeskfaarsvrk")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["intervalKodeAntalAnsatte"].alias("kvarbeskfansat")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["sidstOpdateret"].alias("kvarbeskfopdat")
                                 ,virkMetaData["virkmetadata"]["nyesteMaanedsbeskaeftigelse"]["aar"].alias("mndbeskfaar")
                                 ,virkMetaData["virkmetadata"]["nyesteKvartalsbeskaeftigelse"]["kvartal"].alias("mndbeskfkvart")
                                 ,virkMetaData["virkmetadata"]["nyesteMaanedsbeskaeftigelse"]["intervalKodeAntalAarsvaerk"].alias("mndbeskfaarsvrk")
                                 ,virkMetaData["virkmetadata"]["nyesteMaanedsbeskaeftigelse"]["intervalKodeAntalAnsatte"].alias("mndbeskfansat")
                                 ,virkMetaData["virkmetadata"]["nyesteMaanedsbeskaeftigelse"]["sidstOpdateret"].alias("mndbeskfopdat")
                                 ,virkMetaData["virkmetadata"]["nyesteNavn"]["navn"].alias("navn")
                                 ,virkMetaData["virkmetadata"]["nyesteStatus"]["kreditoplysningkode"].alias("kreditoplysningkode")
                                 ,virkMetaData["virkmetadata"]["nyesteStatus"]["statuskode"].alias("statuskode")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["ansvarligDataleverandoer"].alias("ansvarligDataleverandoer")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["kortBeskrivelse"].alias("kortBeskrivelse")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["sidstOpdateret"].alias("formOpdateret")
                                 ,virkMetaData["virkmetadata"]["nyesteVirksomhedsform"]["virksomhedsformkode"].alias("virksomhedsformkode")
                                 ,virkMetaData["virkmetadata"]["sammensatStatus"].alias("sammensatStatus")
                                 ,virkMetaData["virkmetadata"]["stiftelsesDato"].alias("stiftelsesDato")
                                 ,virkMetaData["virkmetadata"]["virkningsDato"].alias("virkningsDato")
                                 )
#virkDataDf.show(truncate=False)

oneColsVirkDf = virkDataDf.rdd.map(lambda x:columnsToList(x)).toDF()
#print type(oneColsVirkDf)
#oneColsVirkDf.show(truncate=False)
#for i in oneColsVirkDf.first():
#    print type(i)

explodedVirkMetaData = oneColsVirkDf.select(explode(oneColsVirkDf["featlist"]).alias("featlist")).distinct()
explodedVirkMetaData.show(truncate=False)
print "explodedVirkMetaData length", len(explodedVirkMetaData.collect())

virkMetaDataDict = (explodedVirkMetaData
                    .rdd
                    .map(lambda l: l["featlist"])
                    .zipWithIndex()
                    .collectAsMap())

broadcastVirkMetaDataDict = sc.broadcast(virkMetaDataDict)
#print virkMetaDataDict

#TEST example to test bag of words generator
#test = [u'hej',u'med',u'dig',u'hej',u'hej']
#testDicts_broadcast = {'hej':0,'med':1,'dig':2}

bagOfWordsUDF = bagOfWordsGenerator(broadcastVirkMetaDataDict)
featuresDF = oneColsVirkDf.select(bagOfWordsUDF(oneColsVirkDf["featlist"]).alias("features"))
#featuresDF.show(truncate=False)



