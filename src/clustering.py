from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def create_spark_session(app_name="pyspark_experiments"):
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()


def read_iris_data(spark, path="resources/iris.csv"):
    data_schema = [
        StructField('Id', IntegerType(), True),
        StructField('SepalLengthCm', DoubleType(), True),
        StructField('SepalWidthCm', DoubleType(), True),
        StructField('PetalLengthCm', DoubleType(), True),
        StructField('PetalWidthCm', DoubleType(), True),
        StructField('Species', StringType(), True),
    ]

    return spark.read.csv(path, sep=',', header=True, schema=StructType(fields=data_schema))


def create_feature_vector(data):
    vector_assembler = VectorAssembler(
        inputCols=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ],
        outputCol="vector",
    )

    return vector_assembler.transform(data)
    

def normalize_data(data):
    standard_scaler = StandardScaler(
        inputCol="vector", outputCol="features",  withStd=True, withMean=True
    )

    standard_scaler_model = standard_scaler.fit(data)
    return standard_scaler_model.transform(data)


def train_kmeans(data):
    evaluator = ClusteringEvaluator()

    for i in range(2, 10):
        kmeans = KMeans(k=i)
        kmeans_model = kmeans.fit(data)
        
        output = kmeans_model.transform(data)
        score = evaluator.evaluate(output)
        
        print(f"Silhouette Score for {i} clusters: {score}")


if __name__ == "__main__":
    show_rows = 3

    spark = create_spark_session()

    data = read_iris_data(spark)
    print("Raw data:")
    data.show(show_rows)

    data = create_feature_vector(data)
    print("Vectors:")
    data.show(show_rows)

    data = normalize_data(data)
    print("Normalize:")
    data.show(show_rows)

    train_kmeans(data)