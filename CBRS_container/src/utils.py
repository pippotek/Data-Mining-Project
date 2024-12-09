from pyspark.sql import SparkSession

def create_spark_session():
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark
