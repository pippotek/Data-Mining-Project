from pyspark.sql import SparkSession
from training.train_als import train_als_model, load_training_data
from configs.als_configs import ALS_CONFIG
from utilities.logger import get_logger
from data_management.data_utils import load_and_prepare_mind_dataset, preprocess_behaviors_mind

logger = get_logger(name="ALS_Run_Train", log_file="logs/run_train_als.log")

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("ALS_Training") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.cores", "4") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
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