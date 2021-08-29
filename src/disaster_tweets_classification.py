# Spatrk session
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("pyspark_experiments").getOrCreate()


# Read the data
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

data_schema = [
    StructField('id', IntegerType(), True),
    StructField('keyword', StringType(), True),
    StructField('location', StringType(), True),
    StructField('text', StringType(), True),
    StructField('target', DoubleType(), True),
]

data = spark.read.csv(
    "resources/disaster_tweets.csv", sep=',', header=True, schema=StructType(fields=data_schema)
)
print("Raw data:")
data.show(5)


# Filter columns, drop nans
data = data.select("text", "target").na.drop()
print("Filtered data:")
data.show(5)


# Use tf-idf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_data = tokenizer.transform(data)

hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=10000)
featurized_data = hashing_tf.transform(words_data)

idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=20)
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

rescaled_data = rescaled_data.select("target", "features")
print("tf-idf:")
rescaled_data.show(5)


# Splitting
(train_data, valid_data) = rescaled_data.randomSplit([0.7, 0.3], seed=100)
print(f"Train Dataset Count: {str(train_data.count())}")
print(f"Valid Dataset Count:  {str(valid_data.count())}")


# Modeling
from pyspark.ml.classification import LogisticRegression

logistic_regression_model = LogisticRegression(
    maxIter=20, regParam=0.3, elasticNetParam=0, labelCol="target"
)
logistic_regression_model = logistic_regression_model.fit(train_data)

train_predictions = logistic_regression_model.transform(train_data)
valid_predictions = logistic_regression_model.transform(valid_data)


# Eval
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="target")
accuracy = evaluator.evaluate(train_predictions)
print(f"Train Accuracy: {accuracy}")
accuracy = evaluator.evaluate(valid_predictions)
print(f"Valid Accuracy: {accuracy}")