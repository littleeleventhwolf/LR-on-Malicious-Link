from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.feature import RegexTokenizer, HashingTF, 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# schema for dataframe
schema = StructType([StructField("url", StringType()), StructField("label", IntegerType())])
# read the data to generate a dataframe
df = spark.read.schema(schema).csv("file:///home/hadoop/Documents/LR-on-Malicious-Link/data/datacp.csv", header=True)

# string regex tokenizer
tokenizer = RegexTokenizer(inputCol="url", outputCol="words", pattern="/")
# hashing term frequency
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=20)
# inverse document frequency
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

# logistic tregression
lr = LogisticRegression(maxIter=100, regParam=0.001)

# add stages to pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

# train the model
model = pipeline.fit(df)

