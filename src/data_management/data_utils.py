from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, when
from src.utilities.db_utils import read_from_db, write_to_db
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_behaviors(config, spark):
    """
    Preprocesses the interactions table to generate a utility matrix and saves it to the database.
    
    Args:
        config (dict): Configuration dictionary with database connection details.
        spark (SparkSession): Spark session for executing operations.
    """
    logging.info("Starting to preprocess interactions and construct the utility matrix.")

    # Load interactions table from OracleDB
    logging.info("Loading interactions table from OracleDB...")
    interactions_query = "SELECT userid, displayed FROM interactions"
    interactions_df = read_from_db(spark, config, interactions_query)
    logging.info("Interactions table loaded successfully.")

    # Explode displayed articles into individual rows
    logging.info("Exploding displayed articles into individual rows...")
    interactions_df = interactions_df.withColumn("articleid", explode(split(col("displayed"), " ")))

    # Extract clicked information (-1 for clicked, -0 for not clicked)
    logging.info("Extracting clicked information from article IDs...")
    interactions_df = interactions_df.withColumn("clicked", when(col("articleid").endswith("-1"), 1).otherwise(0))
    interactions_df = interactions_df.withColumn("articleid", split(col("articleid"), "-")[0])

    logging.info("Initial DataFrame schema:")
    interactions_df.printSchema()

    # Pivot the table to create a utility matrix
    logging.info("Pivoting the table to create the utility matrix...")
    utility_matrix = interactions_df.groupBy("userid").pivot("articleid").sum("clicked").fillna(0)

    logging.info("Utility matrix constructed successfully.")

    # Write the utility matrix back to OracleDB
    logging.info("Writing the utility matrix to OracleDB...")
    write_to_db(utility_matrix, config, "utility_matrix_user_item", mode="overwrite")
    logging.info("Utility matrix successfully written to OracleDB!")









def preprocess_behaviors(config, spark):
    """
    Preprocesses the interactions table to generate a utility matrix and saves it to the database.
    
    Args:
        config (dict): Configuration dictionary with database connection details.
        spark (SparkSession): Spark session for executing operations.
    """
    # Step 1: Load interactions table from OracleDB
    interactions_query = "SELECT userid, displayed FROM interactions"
    interactions_df = read_from_db(spark, config, interactions_query)

    # Step 2: Explode displayed articles into individual rows
    interactions_df = interactions_df.withColumn("articleid", explode(split(col("displayed"), " ")))

    # Step 3: Extract clicked information (-1 for clicked, -0 for not clicked)
    
    interactions_df = interactions_df.withColumn("clicked", when(col("articleid").endswith("-1"), 1).otherwise(0))
    interactions_df = interactions_df.withColumn("articleid", split(col("articleid"), "-")[0])

    print("Initial DataFrame schema:")
    interactions_df.printSchema()
    
    # Step 4: Pivot the table to create a utility matrix
    utility_matrix = interactions_df.groupBy("userid").pivot("articleid").sum("clicked").fillna(0)

    # Step 5: Write the utility matrix back to OracleDB
    write_to_db(utility_matrix, config, "utility_matrix_user_item", mode="overwrite")
    print("Utility matrix successfully written to OracleDB!")