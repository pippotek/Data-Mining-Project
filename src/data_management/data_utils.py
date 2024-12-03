import os
from typing import Dict
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, split, when, lit, rand
from pyspark.sql.window import Window
from utilities.logger import get_logger
from utilities.db_utils import read_from_db, write_to_db
from recommenders.datasets.mind import download_mind, extract_mind
from recommenders.datasets.download_utils import unzip_file

logger = get_logger("DataUtils", log_file="logs/data_utils.log")


# Preprocesses the behaviors table for training Spark's ALS model
def preprocess_behaviors_mind(
    spark: SparkSession, 
    train_path: str, 
    valid_path: str, 
    npratio: int = 4
):
    logger.info(f"Starting to preprocess MIND dataset. Train: {train_path}, Valid: {valid_path}")
    
    def process_behaviors(df):
        impressions_df = df.withColumn("impression", explode(split(col("impressions"), " ")))
        
        # Extract clicked (1) or non-clicked (0) status
        impressions_df = impressions_df.withColumn(
            "clicked",
            when(col("impression").endswith("-1"), lit(1)).otherwise(lit(0))
        ).withColumn(
            "newsId",
            split(col("impression"), "-")[0]
        ).select("userId", "newsId", "clicked")
        
        positive_samples = impressions_df.filter(col("clicked") == 1)
        negative_samples = impressions_df.filter(col("clicked") == 0) \
            .withColumn("rand", rand())
        
        # Select npratio negative samples per positive sample (addressing class imbalance and making the matrix lighter)
        window = Window.partitionBy("userId").orderBy("rand")
        negative_samples = negative_samples.withColumn("rank", F.row_number().over(window)) \
            .filter(col("rank") <= npratio) \
            .drop("rank", "rand")
        
        combined_samples = positive_samples.union(negative_samples)
        
        return combined_samples

    train_behaviors = spark.read.csv(train_path, sep="\t", header=False) \
        .toDF("impressionId", "userId", "timestamp", "click_history", "impressions")
    valid_behaviors = spark.read.csv(valid_path, sep="\t", header=False) \
        .toDF("impressionId", "userId", "timestamp", "click_history", "impressions")
    
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
        logger.info(f"Downloading and extracting the {size} MIND dataset to {dest_path}...")

        train_zip, valid_zip = download_mind(size=size, dest_path=dest_path)
        extract_mind(
            train_zip, valid_zip,
            train_folder=os.path.join(dest_path, "train"),
            valid_folder=os.path.join(dest_path, "valid"),
        )

        # Correcting paths based on your observed extracted structure
        extracted_base_path = os.path.join(dest_path, f"MIND{size}_train.zip/data/mind")
        extracted_train_path = os.path.join(extracted_base_path, "train/behaviors.tsv")
        extracted_valid_path = os.path.join(extracted_base_path, "valid/behaviors.tsv")

        if not (os.path.exists(extracted_train_path) and os.path.exists(extracted_valid_path)):
            raise FileNotFoundError(f"Extracted files not found: {extracted_train_path}, {extracted_valid_path}")

        logger.info(f"MIND dataset prepared successfully. Train: {extracted_train_path}, Valid: {extracted_valid_path}")
        return extracted_train_path, extracted_valid_path

    except Exception as e:
        logger.error(f"An error occurred while downloading or extracting the MIND dataset: {e}")
        raise
