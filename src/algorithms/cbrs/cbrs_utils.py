from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, split, explode, avg, expr, lit, row_number, desc, when, udf,
    pandas_udf, PandasUDFType, collect_list
)
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import Normalizer, BucketedRandomProjectionLSH
import numpy as np
import pandas as pd
import logging

################################################################################
# Example Utility Functions (Minimal Reproduction)
################################################################################

def fetch_data_from_mongo(spark, mongo_uri, db_name, collection_name):
    """
    Dummy function stub for demonstration.
    Replace with your actual implementation that reads from MongoDB.
    """
    # Example: reading from MongoDB using the Spark MongoDB connector
    return spark.read.format("com.mongodb.spark.sql.DefaultSource") \
        .option("uri", f"{mongo_uri}/{db_name}.{collection_name}") \
        .load()

################################################################################
# Begin Actual Example Code
################################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_data(spark, mongo_uri, db_name, news_col, train_col, test_col):
    logger.info("Loading data from MongoDB...")
    news_embeddings_df = fetch_data_from_mongo(spark, mongo_uri, db_name, news_col)
    behaviors_train_df = fetch_data_from_mongo(spark, mongo_uri, db_name, train_col)
    behaviors_test_df = fetch_data_from_mongo(spark, mongo_uri, db_name, test_col)
    logger.info("Data loaded successfully.")
    
    # Debug: Check counts and sample data
    logger.info(f"News Embeddings Count: {news_embeddings_df.count()}")
    news_embeddings_df.show(5, truncate=False)
    
    logger.info(f"Behaviors Train Count: {behaviors_train_df.count()}")
    behaviors_train_df.show(5, truncate=False)
    
    logger.info(f"Behaviors Test Count: {behaviors_test_df.count()}")
    behaviors_test_df.show(5, truncate=False)
    
    return news_embeddings_df, behaviors_train_df, behaviors_test_df

def preprocess_news_embeddings(news_embeddings_df):
    """
    Preprocesses news embeddings by converting embedding strings to arrays of doubles.
    """
    logger.info("Preprocessing news embeddings...")
    preprocessed_df = (
        news_embeddings_df
        .withColumn(
            "embedding",
            expr("transform(split(embedding_string, ','), x -> cast(x as double))")
        )
        .drop("embedding_string")  # Clean up the column no longer needed
    )
    
    # Debug: Check preprocessing results
    logger.info(f"News Embeddings After Preprocessing Count: {preprocessed_df.count()}")
    preprocessed_df.show(5, truncate=False)
    
    return preprocessed_df

@pandas_udf(ArrayType(DoubleType()), PandasUDFType.GROUPED_AGG)
def average_embeddings(embeddings_series):
    """
    Aggregate function that computes the average of a list of embeddings (arrays).
    """
    embeddings = embeddings_series.tolist()
    if embeddings and len(embeddings) > 0:
        return (np.mean(embeddings, axis=0)).tolist()
    return []

def create_user_profiles_with_pandas_udaf(spark, behaviors_train_df, news_embeddings_df):
    """
    Creates user profiles by averaging the embeddings of the news articles they have interacted with.
    """
    logger.info("Creating user profiles...")
    # Suppose behaviors_train_df has columns: [user_id, history]
    user_history_df = (
        behaviors_train_df
        .withColumn("history_item", explode(split(col("history"), " ")))
        .join(news_embeddings_df, col("history_item") == col("news_id"), "left")
        .select("user_id", col("embedding").alias("embedding"))
        .filter(col("embedding").isNotNull())
    )
    
    # Debug: Check counts after join and filter
    user_history_count = user_history_df.count()
    logger.info(f"User History Count After Join and Filter: {user_history_count}")
    user_history_df.show(5, truncate=False)
    
    if user_history_count == 0:
        logger.error("No user history records found after join and filter. Check your data and join conditions.")
        return spark.createDataFrame([], schema="user_id STRING, user_embedding ARRAY<DOUBLE>")
    
    user_profiles_df = (
        user_history_df
        .groupBy("user_id")
        .agg(average_embeddings(col("embedding")).alias("user_embedding"))
    )
    
    # Debug: Check counts after aggregation
    user_profiles_count = user_profiles_df.count()
    logger.info(f"User Profiles Count: {user_profiles_count}")
    user_profiles_df.show(5, truncate=False)
    
    return user_profiles_df

def evaluate_recommendations(recommendations_df, behaviors_test_df):
    """
    Evaluates the recommendations by computing precision@k.
    """
    logger.info("Evaluating recommendations...")
    test_impressions_df = (
        behaviors_test_df
        .withColumn("impression_item", explode(split(col("impressions"), " ")))
        .withColumn("news_id", split(col("impression_item"), "-")[0])
        .withColumn("actual_clicked", when(split(col("impression_item"), "-")[1] == "1", 1).otherwise(0))
        .select("user_id", "news_id", "actual_clicked")
    )
    
    joined_df = recommendations_df.join(test_impressions_df, ["user_id", "news_id"], "left")
    joined_df = joined_df.fillna({"actual_clicked": 0})
    
    precision_df = joined_df.groupBy("user_id").agg(avg("actual_clicked").alias("user_precision"))
    overall_precision = precision_df.agg(avg("user_precision").alias("avg_precision")).collect()[0]["avg_precision"]
    
    return {"precision@k": overall_precision}

def vectorize_arrays(df, array_col, vector_col):
    """
    Converts array<double> columns to MLlib Vector type.
    """
    to_vector_udf = udf(lambda arr: Vectors.dense(arr) if arr else None, VectorUDT())
    return df.withColumn(vector_col, to_vector_udf(col(array_col)))

################################################################################
# Approximate Nearest Neighbor Recommendation Function
################################################################################

def compute_recommendations_ann(user_profiles_df, news_embeddings_df, top_k=10):
    """
    Uses Approximate Nearest Neighbors (LSH) to find the top_k nearest news articles
    for each user based on embeddings.
    """
    logger.info("Computing recommendations using Approximate Nearest Neighbors (LSH)...")
    
    # 1. Vectorize
    user_profiles_vec = vectorize_arrays(user_profiles_df, "user_embedding", "user_vec")
    news_embeddings_vec = vectorize_arrays(news_embeddings_df, "embedding", "news_vec")
    
    # Debug: Check vectorized DataFrames
    logger.info(f"User Profiles Vectorized Count: {user_profiles_vec.count()}")
    user_profiles_vec.show(5, truncate=False)
    
    logger.info(f"News Embeddings Vectorized Count: {news_embeddings_vec.count()}")
    news_embeddings_vec.show(5, truncate=False)
    
    # 2. Normalize both user and news vectors into a *common* column name.
    normalizer_user = Normalizer(inputCol="user_vec", outputCol="features")
    user_profiles_norm = normalizer_user.transform(user_profiles_vec)
    
    normalizer_news = Normalizer(inputCol="news_vec", outputCol="features")
    news_embeddings_norm = normalizer_news.transform(news_embeddings_vec)
    
    # Debug: Check normalized DataFrames
    logger.info(f"User Profiles Normalized Count: {user_profiles_norm.count()}")
    user_profiles_norm.show(5, truncate=False)
    
    logger.info(f"News Embeddings Normalized Count: {news_embeddings_norm.count()}")
    news_embeddings_norm.show(5, truncate=False)
    
    # 3. Create LSH model for approximate nearest neighbors using the 'features' column
    brp = BucketedRandomProjectionLSH(
        inputCol="features",
        outputCol="hashes",
        bucketLength=10.0,  # Adjust based on data
        numHashTables=3      # Adjust based on data
    )
    logger.info("Fitting LSH model...")
    lsh_model = brp.fit(news_embeddings_norm)
    logger.info("LSH model fitted successfully.")
    
    # 4. Approximate similarity join with a large threshold to include all pairs
    threshold = float("inf")  # effectively no threshold
    logger.info("Performing approximate similarity join...")
    similar_df = lsh_model.approxSimilarityJoin(
        datasetA=user_profiles_norm,
        datasetB=news_embeddings_norm,
        threshold=threshold,
        distCol="distCol"
    )
    
    # Debug: Check similar_df count
    similar_count = similar_df.count()
    logger.info(f"Similar DF Count: {similar_count}")
    similar_df.show(5, truncate=False)
    
    if similar_count == 0:
        logger.error("Similarity join resulted in zero records. Check LSH parameters and data integrity.")
        return spark.createDataFrame([], schema="user_id STRING, news_id STRING, distCol DOUBLE, rank LONG, similarity_score DOUBLE")
    
    # 5. Extract relevant columns and rank
    joined = similar_df.select(
        col("datasetA.user_id").alias("user_id"),
        col("datasetB.news_id").alias("news_id"),
        col("distCol")
    )
    
    window = Window.partitionBy("user_id").orderBy(col("distCol"))
    top_k_df = joined.withColumn("rank", row_number().over(window)).filter(col("rank") <= top_k)
    
    # 6. Compute similarity score (e.g., 1 / (1 + distance))
    top_k_df = top_k_df.withColumn("similarity_score", 1 / (1 + col("distCol")))
    
    # Debug: Check top_k_df
    logger.info(f"Top-K Recommendations Count: {top_k_df.count()}")
    top_k_df.show(5, truncate=False)
    
    return top_k_df
