'''
Created on Jul 8, 2016

@author: svanhmic
'''
# HACK!!!
import sys
from numpy import dtype
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import re
from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import regexp_extract,udf,lit
from pyspark.sql.types import ArrayType, StringType
from pyspark.mllib.linalg import SparseVector,Vectors,VectorUDT


sc = SparkContext("local[8]","test")
sqlContext = SQLContext(sc)




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
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector), "v1s type: " +str(type(v1)) +" v2s type: " + str(type(v2))
    assert v1.size == v2.size , "v1s size: " + str(v1.size) + " v2s size: " + str(v2.size)
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

absSubUDF = udf(absSubtract, VectorUDT())

def computeElementViseDistance(df,clusterCenter):
    '''
        Note:
            
        Args:
        
        Return:
    
    '''
    
sisp1 = sparse
s1 = SparseVector(5,[0,1,2],[-1,1,1])
s2 = SparseVector(4,[1,3],[-4,1])
d1 = Vectors.dense([1,0,1,0])
d2 = Vectors.dense([0,1,0,1])

df = sqlContext.createDataFrame([Row(indx=0,v1=s1,v2=s2),Row(indx=1,v1=s2,v2=s2)])
df.show()

df2 = sqlContext.createDataFrame([Row(indx=1,v3=s1,v4=s1)])
print df.dtypes
    

print createDenseVector(s1)
joinedDf = df.join(df2,df["indx"] == df2["indx"])
print joinedDf.dtypes
diffDf = joinedDf.select(absSubUDF(joinedDf["v1"],joinedDf["v3"]))
diffDf.show(truncate=False)

extraDf = df
extraDf.show(truncate=False)
#df = sqlContext.createDataFrame(Row(first_vec=s1,last_vec=s2), ("first_vec", "last_vec"))
#df.show(truncate=False)


#fileStr = "/home/svanhmic/workspace/Python/Erhvervs/data/cvr"
#virksomheder = "/AlleDatavirksomheder.json"

#virkData = sqlContext.read.format("json").load(fileStr+virksomheder) # loads the subsample of virksomheder
#smallSet = virkData.sample(False,0.0001,42) # collect small sample 
#smallSet.write.json(fileStr+"/Testvirksomheder.json",mode="overwrite")