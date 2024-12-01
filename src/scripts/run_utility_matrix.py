import yaml
from pyspark.sql import SparkSession
from src.data_management.data_utils import preprocess_behaviors

def load_config(file_path):
    """Load YAML configuration."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Load config
    config = load_config("src/config.yaml")

    # Create a minimal Spark Session
    spark = SparkSession.builder \
        .appName("UtilityMatrixGeneration") \
        .master("local[*]") \
        .config("spark.driver.memory", "2G") \
        .config("spark.jars", "/mnt/c/Spark/jars/ojdbc17.jar") \
        .config("spark.local.dir", "/mnt/c/SparkTemp") \
        .config("spark.ui.port", "0") \
        .getOrCreate()

    # Run the utility matrix function
    preprocess_behaviors(config, spark)

    # Stop SparkSession
    spark.stop()
