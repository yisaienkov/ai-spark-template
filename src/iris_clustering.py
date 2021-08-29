# Spatrk session
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("pyspark_experiments").getOrCreate()


# Read the data
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

data_schema = [
    StructField('Id', IntegerType(), True),
    StructField('SepalLengthCm', DoubleType(), True),
    StructField('SepalWidthCm', DoubleType(), True),
    StructField('PetalLengthCm', DoubleType(), True),
    StructField('PetalWidthCm', DoubleType(), True),
    StructField('Species', StringType(), True),
]

data = spark.read.csv(
    "resources/iris.csv", sep=',', header=True, schema=StructType(fields=data_schema)
)
print("Raw data:")
data.show(5)


# Transform data to vector
from pyspark.ml.feature import VectorAssembler

assemble = VectorAssembler(
    inputCols=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ],
    outputCol="features",
)

vector_data = assemble.transform(data)
print("Vectors:")
vector_data.show(5)


# Normalize data
from pyspark.ml.feature import StandardScaler

scale = StandardScaler(
    inputCol="features", outputCol="standardized",  withStd=True, withMean=True
)

model_scale = scale.fit(vector_data)
scaled_data = model_scale.transform(vector_data)
print("Normalize:")
scaled_data.show(5)


# Train KMeans
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

silhouette_score = []
evaluator = ClusteringEvaluator(
    predictionCol="prediction", 
    featuresCol="standardized",
    metricName="silhouette", 
    distanceMeasure="squaredEuclidean"
)

for i in range(2, 10):
    model_kmeans = KMeans(featuresCol="standardized", k=i)
    model_kmeans = model_kmeans.fit(scaled_data)
    
    output = model_kmeans.transform(scaled_data)
    
    score = evaluator.evaluate(output)
    
    silhouette_score.append(score)
    
    print(f"Silhouette Score for {i} clusters: {score}")