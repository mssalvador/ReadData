'''
Created on Aug 16, 2016

This program extracts and cleans metadata

@author: svanhmic
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StringType
from pyspark.sql import functions as F
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.feature import StringIndexer,StringIndexerModel

fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cdata"
hdfsfiles = "hdfs://biml-odin/user/svanhmic/data/CVR"
virksomheder = "/c1p_virksomhed.json"
alleVirksomheder = "/AlleDatavirksomheder.json"
produktionsenhed = "/c1p_produktionsenhed.json"
deltager = "/c1p_deltager.json"
onePercent = "/c1p.json"



def columnsToList(row):
    """ Converts a flat dataframe with columns to a data frame with one column with lists as rows.
        
        Note:
        
        
        Args: 
            row, a row containing data in columns
        
        Returns:
            A row containing one column containing a list, where each element in the list is an column from the input row
    
    """
    featlist=[None if x == None else x for x in row]
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
        
    return Vectors.sparse(lenOfDicts,dicts)
    
def bagOfWordsGenerator(broadcastDictOfWords):
    """ Generates UDF on bag of words from a list of raw words and a dictionary
    
    Note:
        
    Args:
        broadcastDictOfWords 
    Returns:
       UDF which can be used in dataframe select statement
    """
    
    leng = len(broadcastDictOfWords.value)
    return F.udf(lambda x: bagOfWords(x,broadcastDictOfWords,leng),VectorUDT())


columnsToListUDF = F.udf(columnsToList,StringType())
    
if __name__ == '__main__':
    spark = (SparkSession
             .builder
             .appName("createMetaData")
             .getOrCreate())
    
    virkData = (spark
                .read
                .format("json")
                .load(fileStr+alleVirksomheder)) # loads the subsample of virksomheder
    #virkData.printSchema()
    
    #Used when we want the raw data to be parsed
    #virkData = (rawData
    #            .where(rawData["_source"]["Vrvirksomhed"].isNotNull())
    #            .select(rawData["_source"]["Vrvirksomhed"].alias("virksomhed"))
    #            )
    virkMetaData = (virkData
                    .select(virkData["virksomhed"]["cvrNummer"].alias("cvrNummer"),virkData["virksomhed"]["virksomhedMetadata"].alias("virkmetadata"))
                    .cache()) #exectracts metadata
    #virkNulldf = virkMetaData.select(virkMetaData["virkmetadata"].isNull().alias("isnull"),virkMetaData["virkmetadata"])
    #virkMetaData.printSchema()
    
    #normalize the entire metadataset
    virkDataDf = virkMetaData.select(virkMetaData["cvrNummer"]
                                     ,virkMetaData["virkmetadata"]["antalPenheder"].alias("nPenheder")
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
                                     ,F.explode(virkMetaData["virkmetadata"]["nyesteKontaktoplysninger"]).alias("kontakt")
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
    ##virkDataDf.show()
    takeOutMetaDataCols = ["kvarbeskfaar","kvarbeskfkvart","kvarbeskfaarsvrk"
                           ,"kvarbeskfansat","kvarbeskfopdat","mndbeskfaar","mndbeskfkvart"
                           ,"mndbeskfaarsvrk","mndbeskfansat","mndbeskfopdat","aarsVirkDataDf"
                           ,"hovedbranchetekst","formOpdateret","kommunenavn","branchetekst1"
                           ,"branchetekst2","branchetekst3","by","navn","kontakt","vejnavn"]
    aarsVirkDataCols = [x for x in virkDataDf.columns if x not in takeOutMetaDataCols] 
    ##print(aarsVirkDataCols)
    #virkDataDf.printSchema()
    
    aarsVirkDataDf = (virkDataDf
                      .filter(virkDataDf["kvarbeskfaar"].isNull() & virkDataDf["mndbeskfaar"].isNull())
                      .select([F.col(x) for x in aarsVirkDataCols])
                      .fillna("ukendt",["by","landkode","vejnavn","navn","ansvarligDataleverandoer","kortBeskrivelse","sammensatStatus"]))
    #aarsVirkDataDf.show(truncate=False)
    #print(virkDataDf.count())
    #print(aarsVirkDataDf.count())
    
    #Transform kort beskrivelse 
    beskrivelseIndexer = StringIndexer(inputCol="kortBeskrivelse",outputCol="indexKortBeskrivelse")
    beskrivelseIndexerModel = beskrivelseIndexer.fit(aarsVirkDataDf)
    beskrivelseDf = beskrivelseIndexerModel.transform(aarsVirkDataDf)
    
    #Transform ansvarligDataleverandoer
    dataleverandorIndexer = StringIndexer(inputCol="ansvarligDataleverandoer",outputCol="indexKortDataleverandoer")
    dataleverandorIndexerModel = dataleverandorIndexer.fit(beskrivelseDf)
    datalevDf = dataleverandorIndexerModel.transform(beskrivelseDf)
    
    #Transform landekode
    landIndexer = StringIndexer(inputCol="landkode",outputCol="indexLandkode")
    landIndexerModel = landIndexer.fit(datalevDf)
    landDf = landIndexerModel.transform(datalevDf)
    
    landDf.show(truncate=False)
    
    ##explodedVirkMetaData = oneColsVirkDf.select(explode(oneColsVirkDf["featlist"]).alias("featlist")).distinct()
    #explodedVirkMetaData.show(truncate=False)
    ##print("explodedVirkMetaData length", len(explodedVirkMetaData.collect()))
    
    ##virkMetaDataDict = (explodedVirkMetaData
    ##                    .rdd
    ##                    .map(lambda l: l["featlist"])
    ##                    .zipWithIndex()
    ##                    .collectAsMap())
    
    ##broadcastVirkMetaDataDict = sc.broadcast(virkMetaDataDict)
    #print virkMetaDataDict
    
    #TEST example to test bag of words generator
    #test = [u'hej',u'med',u'dig',u'hej',u'hej']
    #testDicts_broadcast = {'hej':0,'med':1,'dig':2}
    
    ##bagOfWordsUDF = bagOfWordsGenerator(broadcastVirkMetaDataDict)
    ##featuresDF = oneColsVirkDf.select(bagOfWordsUDF(oneColsVirkDf["featlist"]).alias("features"))
    #normedFeatures = (featuresDF
    #                  .rdd
    #                  .map(lambda fet: (float(fet["features"].norm(2)),fet["features"]))
    #                  .toDF(["norm","features"])) # computes the norm for each vector
    
    ##Rmatrix = RowMatrix(featuresDF.rdd.map(lambda row: row["features"]))
    
    ##IndexedVectors = featuresDF.rdd.map(lambda row: row["features"]).zipWithIndex().map(lambda x: IndexedRow(x[1],x[0]))
    ##IndexedRowMat = IndexedRowMatrix(IndexedVectors)
    ##CoordinateMat = IndexedRowMat.toCoordinateMatrix()
    ##transposed_CoordinateMat = CoordinateMat.transpose()
    ##transposed_IndexRowMat = transposed_CoordinateMat.toIndexedRowMatrix()
    ##columnSims = transposed_IndexRowMat.columnSimilarities().toIndexedRowMatrix()
    
    ##columSimsRDD = columnSims.rows
    ##print(type(columSimsRDD))
    ##print("cosine similarities rows: ",columnSims.rows)
    ##print("cosine similarities columns: ",columnSims.columns)
    #print cosSimiliarity.numCols()
    #print cosSimiliarity.numRows()
    #print type(cosSimiliarity)
    #rows = sc.parallelize([[1, 2], [1, 5]])
    #print cosSimiliarity.toRowMatrix().rows.take(10)
    #print rows.collect()
    #print mat
    
    #print featuresDF.rdd.take(1)
    
    #columnStatistics = Statistics.colStats(featuresDF.rdd.map(lambda x: x["features"]))
    #featuresDF.show(truncate=False)



