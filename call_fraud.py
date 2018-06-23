from pyspark.context import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import *
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.ml.classification import LogisticRegression

# Grab a handle to the Spark runtime engine

sc = SparkContext.getOrCreate()

# Get access to the SparkContext

sqlContext = SQLContext(sc)

# Load our datafile into memory

schema = StructType([
        StructField("callhour", IntegerType(), True),
        StructField("accountcode", IntegerType(), True),
        StructField("src", IntegerType(), True),
        StructField("dst", FloatType(), True),
        StructField("dst_number_len", IntegerType(), True),
        StructField("lastapp", IntegerType(), True),
        StructField("fraud", IntegerType(), True)
])

df = sqlContext.read.load ("file:./cdr_trainx_min2.csv",
                                format='com.databricks.spark.csv',
                                header='true',
                                schema=schema)

# Display our datafile

df.show()

#Filter out destination numbers where the value is >= 10
#Basically, we don't care about internal calls between customer extensions

df = df.filter(df.dst_number_len >= 10)

df.show()

# Show Schema
df.printSchema()

# Define Features

featureColumns = ["callhour", "accountcode", "src", "dst", "dst_number_len"]
label = ["fraud"]

assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")
transformed = assembler.transform(df)

transformed.show()

# Split the 80% of the data into training data and 20% into testing data

(trainingData, testData) = transformed.randomSplit([0.8,0.2])

print("The size of the training data is %s rows" % trainingData.count())
print("The size of the testing data is %s rows" % testData.count())


# Training the model

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="fraud", featuresCol="features", maxIter=15)

# Train model with Training Data
lrModel = lr.fit(trainingData)


# Predict

predictions = lrModel.transform(testData)
predictions.printSchema()

selected = predictions.select("fraud", "prediction", "probability", "callhour", "src","dst")
selected.filter(predictions.fraud == 1).show()

