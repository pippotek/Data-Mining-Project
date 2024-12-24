from src.algorithms.cbrs.cbrs_utils import * 
from src.utilities.data_utils import * 
from src.algorithms.cbrs.clean_embed import main_embedding
from cbrs_utils import *
import logging
import pandas as pd
import numpy as np

from pymongo.errors import BulkWriteError
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, FloatType, DoubleType
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
from pyspark.ml.linalg import VectorUDT
from src.utilities.data_utils import fetch_data_from_mongo
import pyspark.sql.functions as F
from pyspark.ml.feature import PCA
from pyspark.sql.functions import (
    avg,
    broadcast,
    col,
    collect_list,
    desc,
    explode,
    expr,
    isnan,
    lit,
    pandas_udf,
    PandasUDFType,
    rank,
    row_number,
    split,
    transform,
    udf,
    when
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Content-Based Recommendation System Pipeline")

    try:
        # MongoDB Configuration
        MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
        DATABASE_NAME = "mind_news"
        news_embeddings_collection = "news_combined_embeddings"
        behaviors_train_collection = "behaviors_train"
        behaviors_test_collection = "behaviors_valid"
        RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"

        # Initialize Spark Session with Optimized Configurations
        spark = (SparkSession.builder
                .appName("Combine News and Generate Embeddings")
                .master("local[*]")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "2000M")
                .config("spark.driver.maxResultSize", "0")
                .config("spark.jars.packages",
                        "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                        "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
                .config("spark.mongodb.read.connection.uri", MONGO_URI)
                .config("spark.mongodb.write.connection.uri", MONGO_URI)
                .config("spark.mongodb.output.writeConcern.w", "1")
                .getOrCreate())
        logger.info("Spark Session initialized.")
        
        # If you have a main_embedding function, ensure it's correctly defined and called
        main_embedding(spark)

        # Set Spark Log Level to WARN to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")

        # Load Data
        news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
            spark,
            MONGO_URI,
            DATABASE_NAME,
            news_embeddings_collection,
            behaviors_train_collection,
            behaviors_test_collection
        )
        logger.info("Data loaded successfully.")

        # Preprocess News Embeddings
        news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)
        logger.info("News embeddings preprocessed.")
        # Optional: Preview Data
        # news_embeddings_df.show(2, truncate=True)

        # Create User Profiles
        user_profiles_df = create_user_profiles_with_pandas_udaf(behaviors_train_df, news_embeddings_df)
        # user_profiles_df = user_profiles_df.persist(StorageLevel.MEMORY_AND_DISK)
        # Optional: Preview Data
        # user_profiles_df.show(2, truncate=True)
        logger.info("User profiles created and persisted.")

        # Compute Recommendations using ANN
        top_k = 10  # Set your desired top_k
        recommendations_df = compute_recommendations_ann(
            user_profiles_df,
            news_embeddings_df,
            top_k=top_k
        )
        recommendations_df = recommendations_df.persist(StorageLevel.MEMORY_AND_DISK)
        logger.info("Recommendations computed and persisted.")

        ###############################################################
        # Handle Duplicate '_id' Columns if Any
        if "_id" in recommendations_df.columns:
            # It's common for MongoDB to add an '_id' field. If it already exists in data, consider renaming or dropping.
            recommendations_df = recommendations_df.drop("_id")
            logger.info("Duplicate '_id' column dropped.")

        # Select Relevant Columns
        filtered_recommendations_df = recommendations_df.select(
            "user_id", "news_id", "similarity_score", "rank"
        )

        ###############################################################
        # Optimize Number of Partitions Before Writing
        # Reduce the number of partitions to avoid overwhelming MongoDB
        #optimized_recommendations_df = filtered_recommendations_df.coalesce(50)  # Adjust based on your cluster and MongoDB capacity
        # logger.info("DataFrame repartitioned to 50 partitions for optimized writing.")

        ###############################################################
        
        (recommendations_df
            .write
            .format("mongodb")
            .mode("append")
            .option("database", DATABASE_NAME)
            .option("collection", RECOMMENDATIONS_COLLECTION)
            .option("spark.mongodb.output.batchSize", "1000")  # Adjust batch size as needed
            .save())

        # Optional: Evaluate Recommendations
        # Uncomment the following lines if you wish to evaluate
        #metrics = evaluate_recommendations(recommendations_df, behaviors_test_df)
        #logger.info(f"Evaluation Metrics: {metrics}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Stop Spark Session
        spark.stop()
        logger.info("Spark Session stopped.")
        logger.info("Content-Based Recommendation System Pipeline completed.")

if __name__ == "__main__":
    main()