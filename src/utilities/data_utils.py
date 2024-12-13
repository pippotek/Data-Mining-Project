import os
from typing import Dict
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, split, when, lit, rand
from pyspark.sql.window import Window
from utilities.logger import get_logger
from recommenders.datasets.mind import download_mind, extract_mind
from recommenders.datasets.download_utils import unzip_file

logger = get_logger("DataUtils", log_file="logs/data_utils.log")


from pyspark.sql import SparkSession

def fetch_data_from_mongo(spark: SparkSession, uri: str, db_name: str, collection_name: str):
    """
    Fetch data from a MongoDB collection into a PySpark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        A SparkSession object, already configured to use the Mongo Spark connector.
    uri : str
        The MongoDB connection URI (e.g. "mongodb://user:password@mongodb:27017").
    db_name : str
        The name of the MongoDB database.
    collection_name : str
        The name of the collection to read from.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark DataFrame containing the data from the specified MongoDB collection.
    """
    df = (spark.read
               .format("mongodb")
               .option("uri", uri)
               .option("database", db_name)
               .option("collection", collection_name)
               .load())
    return df

def preprocess_behaviors_mind(
    spark: SparkSession, 
    train_behaviors_df, 
    valid_behaviors_df, 
    npratio: int = 4
):
    logger.info("Starting to preprocess MIND dataset from DataFrames.")
    
    def process_behaviors(df):
        # df expected to have: userId, impressions
        
        impressions_df = df.withColumn("impression", explode(split(col("impressions"), " ")))
    
        # Extract clicked (1) or non-clicked (0)
        impressions_df = impressions_df.withColumn(
            "clicked",
            when(col("impression").endswith("-1"), lit(1)).otherwise(lit(0))
        ).withColumn(
            "newsId",
            split(col("impression"), "-")[0]
        ).select("userId", "newsId", "clicked")
    
        impressions_df = impressions_df.withColumn("userId", F.regexp_replace(col("userId"), "^U", "").cast("int"))
        impressions_df = impressions_df.withColumn("newsId", F.regexp_replace(col("newsId"), "^N", "").cast("int"))
    
        impressions_df = impressions_df.dropna(subset=["userId", "newsId", "clicked"])
        
        positive_samples = impressions_df.filter(col("clicked") == 1)
        negative_samples = impressions_df.filter(col("clicked") == 0).withColumn("rand", rand())
    
        # Select npratio negative samples per positive sample
        window = Window.partitionBy("userId").orderBy("rand")
        negative_samples = negative_samples.withColumn("rank", F.row_number().over(window)) \
            .filter(col("rank") <= npratio) \
            .drop("rank", "rand")
    
        combined_samples = positive_samples.union(negative_samples)
        return combined_samples

    # Rename columns to match what process_behaviors expects
    # Original code expects:
    # impressionId, userId, timestamp, click_history, impressions
    train_behaviors = train_behaviors_df.select(
        col("impression_id").alias("impressionId"),
        col("user_id").alias("userId"),
        col("time").alias("timestamp"),
        col("history").alias("click_history"),
        "impressions"
    )
    
    valid_behaviors = valid_behaviors_df.select(
        col("impression_id").alias("impressionId"),
        col("user_id").alias("userId"),
        col("time").alias("timestamp"),
        col("history").alias("click_history"),
        "impressions"
    )

    train_df = process_behaviors(train_behaviors)
    valid_df = process_behaviors(valid_behaviors)

    logger.info("Preprocessing of MIND dataset completed.")
    return train_df, valid_df


# Loads data from the db and splits it into training and testing datasets.
def load_data_split(spark: SparkSession, config: Dict[str, str], query: str, train_ratio=0.8, seed=47):
    logger.info("Loading data from the database...")
    data = read_from_db(spark, config, query)
    logger.info("Splitting data into training and testing sets...")
    train, test = data.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    logger.info("Data split completed successfully.")
    return train, test


# Downloads, extracts and prepares the MIND dataset from the recommenders library. Returns paths to train and val folders  
def load_and_prepare_mind_dataset(size="demo", dest_path="./data/mind"):
    try:
        logger.info(f"Resolving paths for the {size} MIND dataset...")

        # These paths assume the download was successful
        train_path = f"./MIND{size}_train.zip/data/mind/train/behaviors.tsv"
        valid_path = f"./MIND{size}_train.zip/data/mind/valid/behaviors.tsv"

        if os.path.exists(train_path) and os.path.exists(valid_path):
            logger.info(f"Dataset located. Train: {train_path}, Valid: {valid_path}")
            return train_path, valid_path

        # Download the dataset if not already downloaded
        logger.info(f"Dataset not found. Downloading and extracting {size} MIND dataset to {dest_path}...")
        train_zip, valid_zip = download_mind(size=size, dest_path=dest_path)
        logger.info(f"Downloaded dataset zips: {train_zip}, {valid_zip}")

        # Extract the dataset
        logger.info("Extracting dataset...")
        extract_mind(
            train_zip, valid_zip,
            train_folder=os.path.join(dest_path, "train"),
            valid_folder=os.path.join(dest_path, "valid"),
            clean_zip_file=False  # Ensure zip files are not deleted for debugging
        )

        if os.path.exists(train_path) and os.path.exists(valid_path):
            logger.info(f"Dataset prepared successfully. Train: {train_path}, Valid: {valid_path}")
            return train_path, valid_path

        raise FileNotFoundError(f"Extracted files not found in expected paths: Train: {train_path}, Valid: {valid_path}")

    except Exception as e:
        logger.error(f"An error occurred while preparing the MIND dataset: {e}")
        raise    
