from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType

def load_news_data(spark, news_path):
    """
    Load the news data from a TSV file into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.
    news_path : str
        The path to the news TSV file.

    Returns
    -------
    DataFrame
        A Spark DataFrame containing the news data.
    """
    schema = "NewsID STRING, Category STRING, Subcategory STRING, Title STRING, Abstract STRING, URL STRING, TitleEntities STRING, AbstractEntities STRING"
    news_df = spark.read.csv(
        news_path,
        sep="\t",
        schema=schema,
        header=False
    )
    return news_df

def load_behaviors_data(spark, behaviors_path):
    """
    Load the behaviors data from a TSV file into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.
    behaviors_path : str
        The path to the behaviors TSV file.

    Returns
    -------
    DataFrame
        A Spark DataFrame containing the behaviors data.
    """
    behaviors_schema = StructType([
        StructField("ImpressionID", StringType(), True),
        StructField("UserID", StringType(), True),
        StructField("Time", StringType(), True),
        StructField("History", StringType(), True),
        StructField("Impressions", StringType(), True)
    ])

    behaviors_df = spark.read.csv(
        behaviors_path,
        sep="\t",
        schema=behaviors_schema,
        header=False
    )
    return behaviors_df
