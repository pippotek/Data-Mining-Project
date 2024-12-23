from src.algorithms.cbrs.cbrs_utils import * 
from src.utilities.data_utils import * 
from src.algorithms.cbrs.clean_embed import main_embedding
import logging
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from cbrs_utils import *
from pyspark.sql.functions import col, isnan



# Content-Based Recommender System for News using PySpark with Optimized MongoDB Connector

from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession, DataFrame
from src.utilities.data_utils import fetch_data_from_mongo  # Ensure this utility is implemented
from pyspark.sql.functions import split, explode, when, lit, udf, desc, col, row_number
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import logging
from pyspark.sql.functions import expr, broadcast
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.storagelevel import StorageLevel

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)






from pyspark.sql.types import ArrayType, FloatType, DoubleType
from pyspark.sql import SparkSession, DataFrame
from src.utilities.data_utils import fetch_data_from_mongo
from pyspark.sql.functions import split, explode, avg
from pyspark.sql.functions import explode, split, when, lit, udf, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import col, collect_list
import pandas as pd
from pyspark.sql.functions import rank
import numpy as np
import logging
from pyspark.sql.functions import expr
from pyspark.sql.functions import split, transform, col
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F
from pymongo.errors import BulkWriteError
from pyspark.sql.functions import col, explode, split, when, lit, row_number, desc


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

        # Initialize Spark Session
        spark = (SparkSession.builder
                 .appName("ContentBasedRecSys")
                 .master("local[*]")  # Adjust based on your cluster
                 .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                 .config("spark.kryoserializer.buffer.max", "1g")
                 .config("spark.sql.shuffle.partitions", "400")  # Adjust based on your data size
                 .config("spark.driver.maxResultSize", "8g")
                 .config("spark.memory.fraction", "0.8")
                 .config("spark.memory.storageFraction", "0.3")
                 .config("spark.jars.packages",
                         "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                         "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
                 .config("spark.mongodb.read.connection.uri", MONGO_URI)
                 .config("spark.mongodb.write.connection.uri", MONGO_URI)
                 .config("spark.mongodb.output.writeConcern.w", "1")
                 .config("spark.mongodb.output.batchSize", "1000")  # Adjust batch size as needed
                 .getOrCreate())


        logger.info("Spark Session initialized.")

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

        # Create User Profiles
        user_profiles_df = create_user_profiles_with_pandas_udaf(behaviors_train_df, news_embeddings_df)
        user_profiles_df = user_profiles_df.persist(StorageLevel.MEMORY_AND_DISK)
        user_profiles_df.limit(5).show()
        logger.info("User profiles created and persisted.")

        # Compute recommendations on test data
        recommendations_df = compute_recommendations(behaviors_test_df, news_embeddings_df, user_profiles_df, top_k=5)
        recommendations_df = recommendations_df.persist(StorageLevel.MEMORY_AND_DISK)
        logger.info("Recommendations computed.")

        write_to_mongodb(recommendations_df, MONGO_URI, DATABASE_NAME, RECOMMENDATIONS_COLLECTION)
        logger.info(f"Recommendations written to {RECOMMENDATIONS_COLLECTION} collection in MongoDB.")

        Optional: Evaluate Recommendations
        # Uncomment the following lines if you wish to evaluate
        # metrics = evaluate_recommendations(recommendations_df, behaviors_test_df)
        # logger.info(f"Evaluation Metrics: {metrics}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Stop Spark Session
        spark.stop()
        logger.info("Spark Session stopped.")
        logger.info("Content-Based Recommendation System Pipeline completed.")

if __name__ == "__main__":
    main()



# def main():

#     logger.info("Starting Content-Based Recommendation System Pipeline")

#     try:
#         # MongoDB Configuration
#         MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
#         DATABASE_NAME = "mind_news"
#         news_embeddings_collection = "news_combined_embeddings"
#         behaviors_train_collection = "behaviors_train"
#         behaviors_test_collection = "behaviors_valid"
#         RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"

#         # Initialize Spark Session
#         spark = (SparkSession.builder
#                  .appName("ContentBasedRecSys")
#                  .master("local[*]")  # Adjust based on your cluster
#                  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
#                  .config("spark.kryoserializer.buffer.max", "1g")
#                  .config("spark.sql.shuffle.partitions", "400")  # Adjust based on your data size
#                  .config("spark.driver.maxResultSize", "8g")
#                  .config("spark.memory.fraction", "0.8")
#                  .config("spark.memory.storageFraction", "0.3")
#                  .config("spark.jars.packages",
#                          "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
#                          "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
#                  .config("spark.mongodb.read.connection.uri", MONGO_URI)
#                  .config("spark.mongodb.write.connection.uri", MONGO_URI)
#                  .config("spark.mongodb.output.writeConcern.w", "1")
#                  .config("spark.mongodb.output.batchSize", "1000")  # Adjust batch size as needed
#                  .getOrCreate())

#         logger.info("Spark Session initialized.")

#         # Set Spark Log Level to WARN to reduce verbosity
#         spark.sparkContext.setLogLevel("WARN")

#         # Load Data
#         news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
#             spark,
#             MONGO_URI,
#             DATABASE_NAME,
#             news_embeddings_collection,
#             behaviors_train_collection,
#             behaviors_test_collection
#         )

#         # Preprocess News Embeddings
#         news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)

#         # Create User Profiles
#         user_profiles_df = create_user_profiles_with_pandas_udaf(behaviors_train_df, news_embeddings_df)
#         user_profiles_df = user_profiles_df.persist(StorageLevel.MEMORY_AND_DISK)
#         logger.info("User profiles persisted.")

#         # Compute Recommendations
#         top_k = 10  # Set your desired top_k
#         recommendations_df = compute_recommendations(behaviors_test_df, news_embeddings_df, user_profiles_df, top_k=top_k)

#         ###############################################################
#         # Handle Duplicate '_id' Columns if Any
#         if "_id" in recommendations_df.columns:
#             # It's common for MongoDB to add an '_id' field. If it already exists in data, consider renaming or dropping.
#             recommendations_df = recommendations_df.drop("_id")
#             logger.info("Duplicate '_id' column dropped.")

#         # Select Relevant Columns
#         filtered_recommendations_df = recommendations_df.select(
#             "user_id", "news_id", "clicked", "similarity_score", "rank"
#         )

#         ###############################################################
#         # Optimize Number of Partitions Before Writing
#         # Reduce the number of partitions to avoid overwhelming MongoDB
#         optimized_recommendations_df = filtered_recommendations_df.coalesce(50)  # Adjust based on your cluster and MongoDB capacity
#         logger.info("DataFrame repartitioned to 50 partitions for optimized writing.")

#         ###############################################################
#         # Write Recommendations to MongoDB Using Spark Connector
#         (optimized_recommendations_df
#             .write
#             .format("mongodb")
#             .mode("append")
#             .option("database", DATABASE_NAME)
#             .option("collection", RECOMMENDATIONS_COLLECTION)
#             .option("spark.mongodb.output.batchSize", "1000")  # Adjust batch size as needed
#             .save())
#         logger.info(f"Recommendations written to '{RECOMMENDATIONS_COLLECTION}' collection in MongoDB.")

#         # Optional: Evaluate Recommendations
#         # Uncomment the following lines if you wish to evaluate
#         # metrics = evaluate_recommendations(recommendations_df, behaviors_test_df)
#         # logger.info(f"Evaluation Metrics: {metrics}")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)
#     finally:
#         # Stop Spark Session
#         spark.stop()
#         logger.info("Spark Session stopped.")
#         logger.info("Content-Based Recommendation System Pipeline completed.")

# if __name__ == "__main__":
#     main()

