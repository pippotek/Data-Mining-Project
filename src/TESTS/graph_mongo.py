# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, size
from graphframes import GraphFrame
from setup import load_config

# Load the configuration
config = load_config("src/config.yaml")

# Access specific values with error handling
if config and 'db_connection_string' in config:
    mongodb_uri = config['db_connection_string']
    print(f"Database Connection String: {mongodb_uri}")
else:
    raise ValueError("Failed to load 'db_connection_string' from configuration.")

database = "datamining"  # Replace with your database name
interactions_collection = "interactions"
news_collection = "news"

# Initialize SparkSession with MongoDB and GraphFrames packages
spark = SparkSession.builder \
    .appName("ArticleGraphFromMongoDB") \
    .master("local[*]") \
    .config("spark.ui.enabled", "true") \
    .config("spark.ui.port", "4040") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memoryOverhead", "2g") \
    .config("spark.executor.cores", "4") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.default.parallelism", "200") \
    .config("spark.shuffle.spill", "true") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "4g") \
    .config("spark.mongodb.input.uri", mongodb_uri) \
    .config("spark.mongodb.output.uri", mongodb_uri) \
    .config("spark.jars.packages", ",".join([
        "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1",  
        "graphframes:graphframes:0.8.2-spark3.0-s_2.12"  
    ])) \
    .getOrCreate()




# Set log level to WARN to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# ------------------------------
# Step 1: Read Data from MongoDB
# ------------------------------

# Read interactions data from MongoDB
interactions_df = spark.read \
    .format("mongo") \
    .option("uri", mongodb_uri) \
    .option("database", database) \
    .option("collection", interactions_collection) \
    .load()

# Read news data from MongoDB
news_df = spark.read \
    .format("mongo") \
    .option("uri", mongodb_uri) \
    .option("database", database) \
    .option("collection", news_collection) \
    .load()

# ------------------------------
# Step 2: Process Interactions
# ------------------------------

# Ensure 'articleids' column exists and is an array
if 'articleids' not in interactions_df.columns:
    raise ValueError("'articleids' column not found in interactions collection.")

# Filter out users with fewer than 2 articles
interactions_df = interactions_df.withColumn("article_count", size(col("articleids"))) \
                                 .filter(col("article_count") >= 2)

# Explode the articleids to create individual rows
exploded_df = interactions_df.select(col("_id").alias("userid"), explode(col("articleids")).alias("articleid"))

# Perform a self-join to create all possible unique pairs for each user
pairs_df = exploded_df.alias("a").join(
    exploded_df.alias("b"),
    (col("a.userid") == col("b.userid")) & (col("a.articleid") < col("b.articleid"))
).select(
    col("a.articleid").alias("articleid1"),
    col("b.articleid").alias("articleid2")
)

# ------------------------------
# Step 3: Build Edge List
# ------------------------------

# Group by article pairs and count the occurrences to get edge weights
edge_df = pairs_df.groupBy("articleid1", "articleid2") \
                  .count() \
                  .withColumnRenamed("count", "weight")

# Rename columns for GraphFrame compatibility
edge_df = edge_df.withColumnRenamed("articleid1", "src") \
                 .withColumnRenamed("articleid2", "dst")

# Cast 'src' and 'dst' to string to match 'id' in nodes_df
edge_df = edge_df.withColumn("src", col("src").cast("string")) \
                 .withColumn("dst", col("dst").cast("string"))

# Join with news_df to ensure edges have valid source and destination vertices
# This effectively filters out any edges where 'src' or 'dst' do not exist in news_df
edge_df = edge_df.join(news_df.select(col("_id").alias("src")), on="src", how="inner") \
                 .join(news_df.select(col("_id").alias("dst")), on="dst", how="inner") \
                 .select("src", "dst", "weight")  # Ensure only relevant columns are selected

# ------------------------------
# Step 4: Extract Nodes
# ------------------------------

# Nodes are unique article IDs from the news collection
nodes_df = news_df.select(
    col("_id").alias("id"),
    col("title"),
    col("content")
).distinct()

# Cast 'id' to string to ensure consistency with edge_df
nodes_df = nodes_df.withColumn("id", col("id").cast("string"))

# ------------------------------
# Step 5: Build GraphFrame
# ------------------------------

# Create GraphFrame
g = GraphFrame(nodes_df, edge_df)

print("===== Graph Vertices =====")
g.vertices.show(5, truncate=False)  # Display first 5 vertices

print("===== Graph Edges =====")
g.edges.show(5, truncate=False)     # Display first 5 edges

print("===== Graph Summary =====")
print(f"Number of vertices: {g.vertices.count()}")
print(f"Number of edges: {g.edges.count()}")

# ------------------------------
# Step 6: Apply PageRank
# ------------------------------

# Run PageRank
pagerank_results = g.pageRank(resetProbability=0.15, maxIter=10)

# Extract PageRank scores
pagerank_df = pagerank_results.vertices.select("id", "pagerank", "title")

# Show top 20 articles by PageRank
pagerank_df.orderBy(col("pagerank").desc()).show(20, truncate=False)

# ------------------------------
# Step 7: Save Results (Optional)
# ------------------------------

# Save PageRank scores to MongoDB or other storage
# Example: Save to a collection named 'pagerank_results'
# pagerank_df.write \
#     .format("mongo") \
#     .mode("overwrite") \
#     .option("uri", mongodb_uri) \
#     .option("database", database) \
#     .option("collection", "pagerank_results") \
#     .save()

# Stop SparkSession
spark.stop()
