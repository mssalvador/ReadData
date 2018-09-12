'''
Created on Jan 9, 2017

@author: svanhmic
'''

from pyspark.sql import functions as F
def createNextLayerTable(df,nonExplodedColumns,explodedColumn,*nonExplodedPrefix):
    '''
        The method takes a dataframe and explodes a column of choice such that the contence is more accessible
        
        input 
            - df: data frame with information
            - nonExplodedColumns: List with columns that are "static" the columns are not altered e.g. cvr 
            - explodedColumn: String with the name of the column that gets exploded, note that column must be of an arraytype
            - nonExplodedPrefix: String if values are nested inside a 
        output
            - output: data frame where nonExplodedColumns are unchanged, except for that the values are dupplicated with
            for the explodedColum
    '''
    if len(nonExplodedPrefix) == 1:
        prefixedStr = nonExplodedPrefix[0]+"."
    else:
        prefixedStr = ".".join(nonExplodedPrefix)
    
    if nonExplodedPrefix is None:
        relationsDf = df.select([df[v].alias(v) for v in nonExplodedColumns]+
                            [F.explode(df[explodedColumn]).alias(explodedColumn)])
    else:
        relationsDf = df.select([df[prefixedStr+v].alias(v) for v in nonExplodedColumns]+
                            [F.explode(df[prefixedStr+explodedColumn]).alias(explodedColumn)])
    
    dfSchema = getNextSchemaLayer(relationsDf.schema,explodedColumn)
    return (relationsDf
            .select([relationsDf[u] for u in nonExplodedColumns]
                    +[relationsDf[explodedColumn][v].alias(v) for v in dfSchema]))
    
def getNextSchemaLayer(schema,idx,name="name"):
    schemaDict = schema[idx].jsonValue()
    return list([i[name] for i in schemaDict["type"]["fields"][:]])

def expandSubCols(df,mainColumn,*args):
    '''
    The method expands all subcolumns in the next layer of mainColumn
    
    input:
        df - data frame with data
        mainColumn - the column(s) that contains the subcolumns that should be flattened
        args - extra columns that need to be flattened
    '''
    
    assert mainColumn != "", "mainColumn is empty!"
    columnsToExpand = [mainColumn]+[i for i in args if i in df.columns]
        
    #gets all the schemas for subcols as lists. Flatterend is the total list of 
    schemaList = [(i,getNextSchemaLayer(schema=df.schema,idx=i)) for i in columnsToExpand]
    flat = [(name,subcol) for name,sublist in schemaList for subcol in sublist]
    funcs = [F.col(name+"."+subcol).alias(name+"_"+subcol) for name,subcol in flat ]
    
    
    dfCols = list(filter(lambda x: x not in columnsToExpand,df.columns))
    return df.select(dfCols+ funcs)

if __name__ == '__main__':
    pass