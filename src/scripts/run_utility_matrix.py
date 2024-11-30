import yaml
from pyspark.sql import SparkSession
from ..data_management.data_utils import preprocess_behaviors

def load_config(file_path):
    """Load YAML configuration."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Load config
    config = load_config("src/config.yaml")

    spark = SparkSession.builder \
        .appName("UtilityMatrixGeneration") \
        .config("spark.jars", "C:/Spark/jars/ojdbc8.jar") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.instances", "2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Run the utility matrix function
    preprocess_behaviors(config, spark)

    # Stop SparkSession
    spark.stop()
