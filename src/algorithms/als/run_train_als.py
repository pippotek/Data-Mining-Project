from pyspark.sql import SparkSession
from src.algorithms.als.train_als import train_als_model
from src.utilities.data_utils import  preprocess_behaviors_mind, fetch_data_from_mongo, wait_for_data, write_to_mongodb
from pyspark.sql.functions import col, explode
from src.configs.setup import load_config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    config = load_config('src/configs/config.yaml')
    MONGO_URI = "mongodb://root:example@mongodb:27017"
    # Change this variable to switch data sources: "recommenders" or "db"
    data_source = "db"

    try:
        wait_for_data(
            uri=MONGO_URI,
            db_name="mind_news",
            collection_names=["behaviors_train", "news_train", "behaviors_valid", "news_valid"],
            check_field="_id"
            )
        logger.info("Starting training...")
        spark = (SparkSession.builder
        .appName("ALS_Training")
        .master("local[*]") \
        .config("spark.jars.packages",
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .config("spark.mongodb.read.connection.uri", MONGO_URI) \
        .config("spark.mongodb.write.connection.uri", MONGO_URI) \
        .getOrCreate()
        )

        logger.info("Starting data loading...")
        
        try:
            
            if data_source == "db":
                # New logic: fetch MIND data from MongoDB and preprocess
                MONGO_URI = "mongodb://root:example@mongodb:27017"
                DB_NAME = "mind_news"
                
                # Assume you have two collections: behaviors_train and behaviors_valid
                train_behaviors_df = fetch_data_from_mongo(spark, MONGO_URI, DB_NAME, "behaviors_train")
                valid_behaviors_df = fetch_data_from_mongo(spark, MONGO_URI, DB_NAME, "behaviors_valid")
                
                # preprocess_behaviors_mind version that accepts DataFrames directly
                training_data, validation_data = preprocess_behaviors_mind(spark, train_behaviors_df, valid_behaviors_df, npratio=4)

            else:
                logger.error(f"Unsupported data source: {data_source}")
                raise ValueError(f"Unsupported data source: {data_source}")

            # Train the ALS model using the preprocessed training/validation data
            model_save_path = config['ALS_CONFIG']["model_save_path"]
            logger.info(f"Starting ALS training with data source: {data_source}")
            model = train_als_model(training_data, validation_data, model_save_path)
            reccomendations = model.recommendForAllUsers(numItems= 10)
            # Explode the 'recommendations' column
            exploded_recommendations = reccomendations.select(
                col("userId"),
                explode(col("recommendations")).alias("recommendation")
            )

            # Extract 'itemId' and 'rating' from the 'recommendation' column
            final_recommendations = exploded_recommendations.select(
                col("userId"),
                col("recommendation.newsId").alias("recommendation"),
                col("recommendation.rating").alias("rating"))
            write_to_mongodb(reccomendations, MONGO_URI=MONGO_URI, DATABASE_NAME='mind_news', COLLECTION_NAME='recommendations_als')
        except Exception as e:
            logger.error(f"An error occurred during training: {e}")
        finally:
            try:
                spark.stop()
                logger.info("Spark session stopped.")
            except Exception as e:
                print(f"Error stopping SparkSession: {e}")

    except TimeoutError as e:
        print(f"Error: {e}")
        exit(1)

    

        
