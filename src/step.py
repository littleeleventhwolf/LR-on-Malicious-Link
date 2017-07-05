from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import IntegerType, StringType

schema = StructType([StructField("url", StringType()), StructField("label", IntegerType())])

df = spark.read.schema(schema).csv("file:///home/hadoop/Documents/SVM-on-Malicious-Link/data/datacp.csv", header=True)

from pyspark.ml.feature import RegexTokenizer

tokenizer = RegexTokenizer(inputCol="url", outputCol="words", pattern="/")

from pyspark.ml.feature import HashingTF, IDF

hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=20)

idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=100, regParam=0.001)

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

model = pipeline.fit(df)

