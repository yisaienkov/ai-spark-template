from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def create_spark_session(app_name="pyspark_experiments"):
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()


def read_data(spark, path="resources/disaster_tweets.csv"):
    data_schema = [
        StructField('id', IntegerType(), True),
        StructField('keyword', StringType(), True),
        StructField('location', StringType(), True),
        StructField('text', StringType(), True),
        StructField('target', DoubleType(), True),
    ]

    data = spark.read.csv(path, sep=',', header=True, schema=StructType(fields=data_schema))
    
    # Filter columns, drop nans
    data = data.select("text", "target").na.drop()

    return data


def extract_features(data):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    words_data = tokenizer.transform(data)

    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features")
    featurized_data = hashing_tf.transform(words_data)

    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=20)
    idf_model = idf.fit(featurized_data)
    result_data = idf_model.transform(featurized_data)

    return result_data


def train_logistic_regression(data):
    logistic_regression = LogisticRegression(
        maxIter=20, regParam=0.3, elasticNetParam=0, labelCol="target"
    )
    return logistic_regression.fit(data)


def evaluate_model(model, train_data, valid_data):
    train_predictions = model.transform(train_data)
    valid_predictions = model.transform(valid_data)

    evaluator = BinaryClassificationEvaluator(labelCol="target")
    accuracy = evaluator.evaluate(train_predictions)
    print(f"Train Accuracy: {accuracy}")
    accuracy = evaluator.evaluate(valid_predictions)
    print(f"Valid Accuracy: {accuracy}")


if __name__ == "__main__":
    show_rows = 3

    spark = create_spark_session()

    data = read_data(spark)
    print("Data:")
    data.show(show_rows)

    data = extract_features(data)
    print("Features:")
    data.show(show_rows)
    
    train_data, valid_data = data.randomSplit([0.7, 0.3], seed=100)
    print(f"Train Dataset Count: {str(train_data.count())}")
    print(f"Valid Dataset Count: {str(valid_data.count())}")

    model = train_logistic_regression(train_data)
    evaluate_model(model, train_data, valid_data)

