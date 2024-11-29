from pyspark.sql.dataframe import DataFrame
from typing import Dict
import os
import urllib.request


def download_jdbc_jar(jdbc_jar_url: str, download_path: str):
    """
    Downloads the Oracle JDBC JAR file if it doesn't already exist locally.
    
    Args:
        jdbc_jar_url (str): URL to the Oracle JDBC JAR file.
        download_path (str): Local path to save the JAR file.
    
    Returns:
        str: Path to the downloaded JAR file.
    """
    if os.path.exists(download_path):
        print(f"JDBC JAR already exists at: {download_path}")
    else:
        print(f"Downloading JDBC JAR from {jdbc_jar_url} to {download_path}...")
        try:
            urllib.request.urlretrieve(jdbc_jar_url, download_path)
            print("JDBC JAR downloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to download JDBC JAR: {e}")
    return download_path


def read_from_db(spark, config: Dict[str, str], query: str) -> DataFrame:
    """
    Reads data from an Oracle database into a Spark DataFrame.

    Args:
        spark (SparkSession): Pre-created SparkSession object.
        config (Dict[str, str]): Database configuration dictionary with keys `db_user`, `db_password`, and `db_dsn`.
        query (str): SQL query to execute.

    Returns:
        DataFrame: Spark DataFrame containing the query result.
    """
    url = f"jdbc:oracle:thin:@{config['db_dsn']}?ssl=true"

    
    return spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("dbtable", f"({query})") \
        .option("user", config["db_user"]) \
        .option("password", config["db_password"]) \
        .option("driver", "oracle.jdbc.OracleDriver") \
        .load()


def write_to_db(df: DataFrame, config: Dict[str, str], table_name: str, mode: str = "overwrite"):
    """
    Writes a Spark DataFrame to an Oracle database table.

    Args:
        df (DataFrame): Spark DataFrame to write to the database.
        config (Dict[str, str]): Database configuration dictionary with keys `db_user`, `db_password`, and `db_dsn`.
        table_name (str): Name of the target table in the database.
        mode (str): Save mode for writing the DataFrame (default: "overwrite").

    Returns:
        None
    """
    url = f"jdbc:oracle:thin:@{config['db_dsn']}"
    
    df.write \
        .format("jdbc") \
        .option("url", url) \
        .option("dbtable", table_name) \
        .option("user", config["db_user"]) \
        .option("password", config["db_password"]) \
        .option("driver", "oracle.jdbc.OracleDriver") \
        .mode(mode) \
        .save()
