from pyspark.sql import SparkSession
from training.train_als import train_als_model, load_training_data
from configs.als_configs import ALS_CONFIG
from utilities.logger import get_logger
from data_management.data_utils import load_and_prepare_mind_dataset, preprocess_behaviors_mind

logger = get_logger(name="ALS_Run_Train", log_file="logs/run_train_als.log")

# NOTE: I'VE ADDED A LOT OF SHIT HERE DURING DEBUGGING. WE SHOULD CHECK THIS IN FINAL VERSION.
if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("ALS_Training") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.cores", "6") \
        .config("spark.driver.cores", "6") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.default.parallelism", "16") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer", "64m") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .getOrCreate()       

    # Define the data source
    data_source = "recommenders"  # Change to "db" or "csv" to switch data sources

    logger.info("Starting data loading...")
    try:
        # Load data based on the specified source
        if data_source == "recommenders":
            # Prepare MIND dataset
            train_path, valid_path = load_and_prepare_mind_dataset(size="demo", dest_path="./data/mind")
            training_data, validation_data = preprocess_behaviors_mind(spark, train_path, valid_path, npratio=4)
            #training_data = training_data.limit(1000) #just for debugging purposes, shouldn't be included in final version
        else:
            logger.error(f"Unsupported data source: {data_source}")
            raise ValueError(f"Unsupported data source: {data_source}")

        # Train the ALS model
        model_save_path = ALS_CONFIG["model_save_path"]
        logger.info(f"Starting ALS training with data source: {data_source}")
        train_als_model(training_data, validation_data, model_save_path)

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
    finally:
        spark.stop()
        logger.info("Spark session stopped.")