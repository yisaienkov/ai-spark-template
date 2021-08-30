from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def create_spark_session(app_name="pyspark_experiments"):
    return SparkSession.builder.master("local[*]").appName(app_name).getOrCreate()


def read_data(spark, path="resources/heart_failure_clinical_records_dataset.csv"):
    data_schema = [
        StructField('age', IntegerType(), True),
        StructField('anaemia', IntegerType(), True),
        StructField('creatinine_phosphokinase', IntegerType(), True),
        StructField('diabetes', IntegerType(), True),
        StructField('ejection_fraction', IntegerType(), True),
        StructField('high_blood_pressure', IntegerType(), True),
        StructField('platelets', IntegerType(), True),
        StructField('serum_creatinine', DoubleType(), True),
        StructField('serum_sodium', IntegerType(), True),
        StructField('sex', IntegerType(), True),
        StructField('smoking', IntegerType(), True),
        StructField('time', IntegerType(), True),
        StructField('DEATH_EVENT', IntegerType(), True),
    ]
    data = spark.read.csv(path, sep=',', header=True, schema=StructType(fields=data_schema))
    data = data.na.drop()

    return data


def create_cv_model(train_data):
    vector_assembler = VectorAssembler(
        inputCols=[
            "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", 
            "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", 
            "smoking", "time"
        ],
        outputCol="vector",
    )
    standard_scaler = StandardScaler(
        inputCol="vector", outputCol="features",  withStd=True, withMean=True
    )
    logistic_regression = LogisticRegression(maxIter=10, labelCol="DEATH_EVENT")
    pipeline = Pipeline(stages=[vector_assembler, standard_scaler, logistic_regression])

    param_grid = ParamGridBuilder() \
        .addGrid(standard_scaler.withMean, [False, True]) \
        .addGrid(logistic_regression.regParam, [0.1, 0.01, 0.001]) \
        .build()

    cross_validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=BinaryClassificationEvaluator(labelCol="DEATH_EVENT"),
        numFolds=5,
    )

    return cross_validator.fit(train_data)


def evaluate_model(model, train_data, valid_data):
    train_predictions = model.transform(train_data)
    valid_predictions = model.transform(valid_data)

    evaluator = BinaryClassificationEvaluator(labelCol="DEATH_EVENT")
    accuracy = evaluator.evaluate(train_predictions)
    print(f"Train Accuracy: {accuracy}")
    accuracy = evaluator.evaluate(valid_predictions)
    print(f"Valid Accuracy: {accuracy}")


if __name__ == "__main__":
    show_rows = 3

    spark = create_spark_session()

    data = read_data(spark)
    print("Raw data:")
    data.show(show_rows)

    train_data, valid_data = data.randomSplit([0.7, 0.3], seed=100)
    print(f"Train Dataset Count: {str(train_data.count())}")
    print(f"Valid Dataset Count: {str(valid_data.count())}")

    model = create_cv_model(train_data)
    evaluate_model(model, train_data, valid_data)