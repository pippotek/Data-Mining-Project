remove duplicates: # src/algorithms/cbrs/pipeline.py
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import SparkSession, DataFrame
from src.utilities.data_utils import * 
from pyspark.sql.functions import split, explode, avg
from pyspark.sql.functions import explode, split, when, lit, udf, desc
from pyspark.sql.window import Window
from pyspark.sql.functions import collect_list
from pyspark.sql.functions import pandas_udf, col, collect_list
import pandas as pd
from pyspark.sql.functions import rank
import numpy as np
from pyspark.sql.functions import pandas_udf

import logging
from pyspark.sql.functions import expr
from pyspark.sql.functions import split, transform, col
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, DoubleType
import pandas as pd
from pyspark.ml.linalg import VectorUDT
import numpy as np
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, explode, split, when, lit, row_number, desc


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



# @udf(FloatType())
# def cosine_similarity_udf(u, n):
#     """
#     Compute cosine similarity between two arrays of floats.
#     """
#     if u is None or n is None:
#         return 0.0
#     u_arr = np.array(u, dtype=float)
#     n_arr = np.array(n, dtype=float)
#     norm_u = np.linalg.norm(u_arr)
#     norm_n = np.linalg.norm(n_arr)
#     if norm_u == 0 or norm_n == 0:
#         return 0.0
#     return float(np.dot(u_arr, n_arr) / (norm_u * norm_n))



def compute_recommendations(
    behaviors_df: DataFrame, 
    news_embeddings_df: DataFrame, 
    user_profiles_df: DataFrame, 
    top_k: int = 5
) -> DataFrame:
    """
    Compute content-based recommendations by joining impressions with user profiles and scoring via cosine similarity.
    Returns a DataFrame with user_id, news_id, similarity_score, and rank, filtered to top_k.
    """

    # Parse impressions into individual rows
    impressions_df = (
        behaviors_df
        .withColumn("impression_item", explode(split(col("impressions"), " ")))
        .withColumn("news_id", split(col("impression_item"), "-")[0])
        .withColumn("clicked", split(col("impression_item"), "-")[1].cast("int"))
        .drop("impression_item")
    )

    # Join with news embeddings to get 'embedding'
    impressions_with_embeddings_df = impressions_df.join(news_embeddings_df, on="news_id", how="left")

    # Join with user profiles to get 'user_embedding'
    impressions_with_profiles_df = impressions_with_embeddings_df.join(user_profiles_df, on="user_id", how="left")

    # Compute similarity scores 
    @F.udf("double")
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

    scored_df = impressions_with_profiles_df.withColumn(
        "similarity_score",
        cosine_similarity_udf(col("user_embedding"), col("embedding"))
    )

    # Rank impressions by similarity score for each user
    window = Window.partitionBy("user_id").orderBy(desc("similarity_score"))
    recommendations_df = scored_df.withColumn("rank", row_number().over(window)).filter(col("rank") <= top_k)

    return recommendations_df



def evaluate_recommendations(
    recommendations_df: DataFrame, 
    behaviors_test_df: DataFrame
) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("Evaluating recommendations...")

    rec_selected = recommendations_df.select("user_id", "news_id")

    test_impressions_df = (
        behaviors_test_df
        .withColumn("impression_item", explode(split(col("impressions"), " ")))
        .withColumn("news_id", split(col("impression_item"), "-")[0])
        .withColumn("actual_clicked", when(split(col("impression_item"), "-")[1] == "1", lit(1)).otherwise(lit(0)))
        .select("user_id", "news_id", "actual_clicked")
    )

    joined_df = rec_selected.join(test_impressions_df, on=["user_id", "news_id"], how="left")

    # Fill nulls with 0
    joined_df = joined_df.withColumn("actual_clicked", when(col("actual_clicked").isNull(), 0).otherwise(col("actual_clicked")))

    # Group and compute average precision per user
    precision_df = joined_df.groupBy("user_id").agg(F.avg("actual_clicked").alias("user_precision"))

    # Compute overall precision@k
    #overall_precision = precision_df.agg(F.avg("user_precision").alias("avg_precision")).collect()[0]["avg_precision"]
    result_row = precision_df.agg(F.avg("user_precision").alias("avg_precision")).head(1)
    overall_precision = result_row[0]["avg_precision"] if result_row else None
    
    metrics = {
        "precision@k": overall_precision
    }

    return metrics
