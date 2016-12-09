#    pdfFile = textract.process("/home/svanhmic/workspace/Python/SparkTestProgram/data/ex3.pdf",method="pdftotext",encoding="ascii")
    
#    stringArray = pdfFile.encode("unicode").split()
    #print stringArray
#    pdfRDD = sc.parallelize(stringArray)
#    mappedPDFRDD = pdfRDD.map(lambda words: (words.lower(),1))
#    reducedPdf = mappedPDFRDD.reduceByKey(lambda x,y: x+y)
#    print reducedPdf.collect()
    #