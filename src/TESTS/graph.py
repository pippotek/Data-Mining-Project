from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit, array, split, size, array_distinct

# Define Spark configuration
conf = SparkConf() \
    .setAppName("DataMiningProject") \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "4g") \
    .set("spark.executor.cores", "2") \
    .set("spark.executor.instances", "4") \
    .set("spark.sql.shuffle.partitions", "200")\
    .set("spark.memory.fraction", "0.8") \
    .set("spark.memory.storageFraction", "0.2") \
    .set("spark.shuffle.spill", "true") \
    .set("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .set("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
    .set("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")

# Initialize the Spark session
spark = SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()

# Set the log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")
# Verify Spark session and configuration
print("Spark session started with the following configurations:")
for item in spark.sparkContext.getConf().getAll():
    print(f"{item[0]} = {item[1]}")


# Define schema for interactions TSV file
from pyspark.sql.types import StructType, StructField, StringType

interactions_schema = StructType([
    StructField("session_id", StringType(), True),
    StructField("userid", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("articleids", StringType(), True),
    StructField("additional_data", StringType(), True)
])

# Load interactions data
interactions_df = spark.read.csv(
    "merged_file.tsv",
    sep="\t",
    header=False,
    schema=interactions_schema
)

# Rename columns for clarity
interactions_df = interactions_df.withColumnRenamed("_c0", "session_id") \
                                 .withColumnRenamed("_c1", "userid") \
                                 .withColumnRenamed("_c2", "timestamp") \
                                 .withColumnRenamed("_c3", "articleids") \
                                 .withColumnRenamed("_c4", "additional_data")

# Split 'articleids' into an array of individual article IDs
interactions_df = interactions_df.withColumn("articleids", split(col("articleids"), " "))

# Filter out rows with fewer than 2 articles
interactions_df = interactions_df.withColumn("article_count", size(col("articleids"))) \
                                 .filter(col("article_count") >= 2)

# Explode the array of article IDs into individual rows
exploded_df = interactions_df.withColumn("articleid", explode(col("articleids")))

# Perform a self-join to create all possible pairs within the same session
pairs_df = exploded_df.alias("a").join(
    exploded_df.alias("b"),
    (col("a.session_id") == col("b.session_id")) & (col("a.articleid") < col("b.articleid"))
)

# Select relevant columns and group to count occurrences
edge_df = pairs_df.groupBy("a.articleid", "b.articleid") \
                  .count() \
                  .withColumnRenamed("a.articleid", "articleid1") \
                  .withColumnRenamed("b.articleid", "articleid2") \
                  .withColumnRenamed("count", "weight")

# Display the resulting edge list
edge_df.show(truncate=False)

# Stop SparkSession
spark.stop()
