from src.algorithms.cbrs.cbrs_utils import * 
from src.utilities.data_utils import * 
import logging
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from cbrs_utils import *

def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("Starting Content-Based Recommendation System Pipeline")

    try:
        MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
        DATABASE_NAME = "mind_news"
        news_embeddings_collection = "news_combined_embeddings"
        behaviors_train_collection = "behaviors_train"
        behaviors_test_collection = "behaviors_valid"
        RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"

        spark = (SparkSession.builder
                 .appName("ContentBasedRecSys")
                 .master("local[*]")
                 .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                 .config("spark.kryoserializer.buffer.max", "1g")
                 .config("spark.sql.shuffle.partitions", "400")
                 .config("spark.driver.maxResultSize", "8g")
                 .config("spark.memory.fraction", "0.8")
                 .config("spark.memory.storageFraction", "0.3")
                 .config("spark.jars.packages",
                         "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                         "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
                 .config("spark.mongodb.read.connection.uri", MONGO_URI)
                 .config("spark.mongodb.write.connection.uri", MONGO_URI)
                 .config("spark.mongodb.output.writeConcern.w", "1")
                 .getOrCreate())

        logger.info("Spark Session initialized.")

        # Load data
        news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
            spark,
            MONGO_URI,
            DATABASE_NAME,
            news_embeddings_collection,
            behaviors_train_collection,
            behaviors_test_collection
        )

        logger.info("Data loaded successfully.")

        # Preprocess news embeddings
        news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)
        logger.info("News embeddings preprocessed.")

        # Create user profiles from training data using Pandas UDAF
        user_profiles_df = create_user_profiles_with_pandas_udaf(behaviors_train_df, news_embeddings_df)
        user_profiles_df = user_profiles_df.persist(StorageLevel.MEMORY_AND_DISK)
        logger.info("User profiles created using Pandas UDAF.")

        # Compute recommendations on test data
        recommendations_df = compute_recommendations(behaviors_test_df, news_embeddings_df, user_profiles_df, top_k=5)
        recommendations_df = recommendations_df.persist(StorageLevel.MEMORY_AND_DISK)
        logger.info("Recommendations computed.")
        
        
        
        ###############################################################
        
        # Drop or rename duplicate '_id' column
        if recommendations_df.columns.count("_id") > 1:
            # Drop all '_id' columns to prevent duplication
            recommendations_df = recommendations_df.drop("_id")
            logger.info("Duplicate '_id' column dropped.")

        filtered_recommendations_df = recommendations_df.select(
        "user_id", "news_id", "clicked", "similarity_score", "rank")

        write_to_mongodb(filtered_recommendations_df, MONGO_URI, DATABASE_NAME, RECOMMENDATIONS_COLLECTION)
        logger.info(f"Recommendations written to {RECOMMENDATIONS_COLLECTION} collection in MongoDB.")

        # # Evaluate recommendations
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
