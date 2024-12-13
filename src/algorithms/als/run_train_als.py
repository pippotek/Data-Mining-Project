from pyspark.sql import SparkSession
from src.algorithms.als.train_als import train_als_model
from src.algorithms.als.als_configs import ALS_CONFIG
from src.utilities.logger import get_logger
from src.utilities.data_utils import  preprocess_behaviors_mind, fetch_data_from_mongo, wait_for_data
from pyspark.sql import SparkSession


logger = get_logger(name="ALS_Run_Train", log_file="logs/run_train_als.log")

if __name__ == "__main__":


    MONGO_URI = "mongodb://root:example@mongodb:27017"
    # Change this variable to switch data sources: "recommenders" or "db"
    data_source = "db"

    try:
        wait_for_data(
            uri=MONGO_URI,
            db_name="db_news",
            collection_name="behaviors_train",
            check_field="impressions",  # A field you expect in the collection
        )
        print("Starting training...")
        spark = (SparkSession.builder
        .appName("ALS_Training")
        .master("local[*]") \
        .config("spark.driver.memory", "8G") \
        .config("spark.executor.memory", "8G") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.driver.maxResultSize", "0") \
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
            model_save_path = ALS_CONFIG["model_save_path"]
            logger.info(f"Starting ALS training with data source: {data_source}")
            train_als_model(training_data, validation_data, model_save_path)

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

    

        
