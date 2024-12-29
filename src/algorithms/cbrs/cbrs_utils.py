from pyspark.sql.types import ArrayType, FloatType, DoubleType
from pyspark.sql import SparkSession, DataFrame
from src.utilities.data_utils import * 
from pyspark.sql.functions import split, explode, avg
from pyspark.sql.functions import explode, split, when, lit, udf, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import col, collect_list
import pandas as pd
from pyspark.sql.functions import rank
import numpy as np
import logging
from pyspark.sql.functions import expr
from pyspark.sql.functions import split, transform, col
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.ml.linalg import VectorUDT
import pyspark.sql.functions as F
from pymongo.errors import BulkWriteError
from pyspark.sql.functions import col, explode, split, when, lit, row_number, desc

from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import row_number
from pyspark.sql.window import Window


def load_data(
    spark: SparkSession, 
    mongo_uri: str, 
    db_name: str, 
    news_embeddings_collection: str,
    behaviors_train_collection: str,
    behaviors_test_collection: str
) -> tuple:

    news_embeddings_df = fetch_data_from_mongo(spark, mongo_uri, db_name, news_embeddings_collection)
    behaviors_train_df = fetch_data_from_mongo(spark, mongo_uri, db_name, behaviors_train_collection)
    behaviors_test_df = fetch_data_from_mongo(spark, mongo_uri, db_name, behaviors_test_collection)
    return news_embeddings_df, behaviors_train_df, behaviors_test_df



def preprocess_news_embeddings(news_embeddings_df: DataFrame) -> DataFrame:
    """
    Preprocess news embeddings DataFrame by converting embedding_string to an array of doubles using built-in Spark functions.
    """
    news_embeddings_df = news_embeddings_df.withColumn(
        "embedding",
        expr("transform(split(embedding_string, ','), x -> cast(x as double))")
    ).drop("embedding_string")
    return news_embeddings_df



@pandas_udf(ArrayType(DoubleType()), PandasUDFType.GROUPED_AGG)
def average_embeddings(embeddings_series: pd.Series) -> list:
    """
    Compute the element-wise average of a series of embeddings.
    """
    embeddings = embeddings_series.tolist()
    if not embeddings:
        return []
    sum_embedding = np.sum(embeddings, axis=0)
    avg_embedding = sum_embedding / len(embeddings)
    return avg_embedding.tolist()



def create_user_profiles_with_pandas_udaf(behaviors_train_df: DataFrame, news_embeddings_df: DataFrame) -> DataFrame:
    """
    Create user profiles by averaging embeddings using a Pandas UDAF.
    """
    behaviors_train_alias = behaviors_train_df.alias("bt")
    news_embeddings_alias = news_embeddings_df.alias("ne")

    # Explode user history into individual news items and join with news embeddings
    user_history_df = (
        behaviors_train_alias
        .withColumn("history_item", explode(split(col("bt.history"), " ")))
        .join(
            news_embeddings_alias,
            col("history_item") == col("ne.news_id"),
            how="left"
        )
        .select(col("bt.user_id"), col("ne.embedding").alias("embedding"))
        .filter(col("embedding").isNotNull())
    )

    # Group by user_id and compute average embeddings
    user_profiles_df = user_history_df.groupBy("user_id").agg(
        average_embeddings(col("embedding")).alias("user_embedding")
    )

    return user_profiles_df



def compute_recommendations(
    behaviors_df: DataFrame, 
    news_embeddings_df: DataFrame, 
    user_profiles_df: DataFrame, 
    top_k: int = 10
) -> DataFrame:
    """
    Compute content-based recommendations by joining impressions with user 
    profiles and scoring via cosine similarity. Returns a DataFrame with
    user_id, news_id, similarity_score, and rank, filtered to top_k per user.
    """
    logger.info("Computing recommendations...")

    # 1) Parse impressions into individual rows
    impressions_df = (
        behaviors_df
        .withColumn("impression_item", explode(split(col("impressions"), " ")))
        .withColumn("news_id", split(col("impression_item"), "-")[0])
        .withColumn("clicked", split(col("impression_item"), "-")[1].cast("int"))
        .drop("impression_item")
    )
    logger.info("Impressions parsed.")
    
    # Join with news embeddings
    # 2) Join with news embeddings
    impressions_with_embeddings_df = impressions_df.join(
        news_embeddings_df, on="news_id", how="left"
    )
    logger.info("Joined impressions with news embeddings.")

    # 3) Join with user profiles to get 'user_embedding'
    impressions_with_profiles_df = impressions_with_embeddings_df.join(
        user_profiles_df, on="user_id", how="left"
    )
    logger.info("Joined impressions with user profiles.")

    # 4) Define UDF for cosine similarity
    @udf(DoubleType())
    def cosine_similarity_udf(u, n):
        """
        Compute cosine similarity between two arrays of doubles.
        """
        if u is None or n is None:
            return 0.0
        u_arr = np.array(u, dtype=float)
        n_arr = np.array(n, dtype=float)
        norm_u = np.linalg.norm(u_arr)
        norm_n = np.linalg.norm(n_arr)
        if norm_u == 0 or norm_n == 0:
            return 0.0
        return float(np.dot(u_arr, n_arr) / (norm_u * norm_n))

    # 5) Compute similarity scores
    scored_df = impressions_with_profiles_df.withColumn(
        "similarity_score",
        cosine_similarity_udf(col("user_embedding"), col("embedding"))
    )
    logger.info("Similarity scores computed.")

    # 6) (Optional) Remove entries with similarity_score <= 0 to reduce data size
    scored_df = scored_df.filter(col("similarity_score") > 0)
    logger.info("Filtered out zero similarity scores.")

    # 7) Repartition by user_id (to optimize window operation)
    scored_df = scored_df.repartition("user_id")

    # 8) Rank and filter to top_k per user
    window = Window.partitionBy("user_id").orderBy(desc("similarity_score"))
    recommendations_df = (
        scored_df
        .withColumn("rank", row_number().over(window))
        .filter(col("rank") <= top_k)  # Only keep top_k
    )

    logger.info(f"Top-{top_k} recommendations per user selected.")

    return recommendations_df



def evaluate_recommendations(
    recommendations_df: DataFrame, 
    behaviors_test_df: DataFrame
) -> dict:
    """
    Evaluate the recommendations using precision@k metric.
    """
    logger.info("Evaluating recommendations...")

    rec_selected = recommendations_df.select("user_id", "news_id")

    test_impressions_df = (
        behaviors_test_df
        .withColumn("impression_item", explode(split(col("impressions"), " ")))
        .withColumn("news_id", split(col("impression_item"), "-")[0])
        .withColumn("actual_clicked", when(split(col("impression_item"), "-")[1] == "1", lit(1)).otherwise(lit(0)))
        .select("user_id", "news_id", "actual_clicked")
    )

    logger.info("Test impressions parsed.")

    joined_df = rec_selected.join(test_impressions_df, on=["user_id", "news_id"], how="left")

    # Fill nulls with 0 (no click)
    joined_df = joined_df.withColumn("actual_clicked", when(col("actual_clicked").isNull(), 0).otherwise(col("actual_clicked")))

    logger.info("Joined recommendations with actual clicks.")

    # Compute average precision per user
    precision_df = joined_df.groupBy("user_id").agg(avg("actual_clicked").alias("user_precision"))

    # Compute overall precision@k
    result_row = precision_df.agg(avg("user_precision").alias("avg_precision")).head(1)
    overall_precision = result_row[0]["avg_precision"] if result_row else None

    metrics = {
        "precision@k": overall_precision
    }

    logger.info(f"Evaluation Metrics: {metrics}")

    return metrics


def vectorize_arrays(df: DataFrame, array_col: str, vector_col: str) -> DataFrame:
    """
    Convert an array<double> column into a Spark ML Vector column.
    """
    to_vector_udf = F.udf(lambda arr: Vectors.dense(arr) if arr else None, VectorUDT())
    return df.withColumn(vector_col, to_vector_udf(F.col(array_col)))


def compute_recommendations_ann(
    user_profiles_df: DataFrame,
    news_embeddings_df: DataFrame,
    top_k: int = 10
) -> DataFrame:
    """
    Use Approximate Nearest Neighbors (LSH) to find the top_k nearest news articles
    for each user, based on embeddings.

    Returns a DataFrame with columns:
        - user_id
        - news_id
        - distCol (the approximate distance)
        - rank
        - similarity_score
    """

    # 1) Convert array<double> -> Vector
    user_profiles_vec = vectorize_arrays(user_profiles_df, "user_embedding", "user_vec")
    news_embeddings_vec = vectorize_arrays(news_embeddings_df, "embedding", "news_vec")

    # 2) Normalize the vectors to unit length for cosine similarity
    normalizer = Normalizer(inputCol="user_vec", outputCol="normalized_vec", p=2.0)
    user_profiles_norm = normalizer.transform(user_profiles_vec)

    news_normalizer = Normalizer(inputCol="news_vec", outputCol="normalized_vec", p=2.0)
    news_embeddings_norm = news_normalizer.transform(news_embeddings_vec)

    # 3) Create an LSH model for approximate nearest neighbors
    brp = BucketedRandomProjectionLSH(
        inputCol="normalized_vec",
        outputCol="hashes",
        bucketLength=10.0,      # Tuning parameter
        numHashTables=3          # Tuning parameter
    )
    
    # 4) Fit the LSH model on the users
    lsh_model = brp.fit(user_profiles_norm)

    # 5) approxSimilarityJoin to find top neighbors between users (datasetA) and news (datasetB)
    similar_df = lsh_model.approxSimilarityJoin(
        datasetA=user_profiles_norm,
        datasetB=news_embeddings_norm,
        threshold=float("inf"),   # No distance threshold
        distCol="distCol"         # Name of the distance column
    )

    # 6) Extract user_id and news_id from the struct columns
    joined = similar_df.select(
        F.col("datasetA.user_id").alias("user_id"),
        F.col("datasetB.news_id").alias("news_id"),
        F.col("distCol")
    )

    # 7) Rank results by ascending distance (the smaller the distance, the better)
    window = Window.partitionBy("user_id").orderBy(F.asc("distCol"))
    ranked = joined.withColumn("rank", row_number().over(window))

    # 8) Filter to top_k
    top_k_df = ranked.filter(F.col("rank") <= top_k)

    # 9) Compute a similarity score if desired
    top_k_df = top_k_df.withColumn("similarity_score", 1 / (1 + F.col("distCol")))

    return top_k_df