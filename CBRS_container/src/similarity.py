import numpy as np
from pyspark.sql.functions import udf, collect_list, size
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.sql.functions as F

def cosine_similarity_udf(u_vec, i_vec):
    """
    Compute cosine similarity between two Spark ML vectors.

    Parameters
    ----------
    u_vec : Vector or None
        The user vector.
    i_vec : Vector or None
        The item vector.

    Returns
    -------
    float
        The cosine similarity.
    """
    if u_vec is None or i_vec is None:
        return 0.0
    u_arr = u_vec.toArray()
    i_arr = i_vec.toArray()
    norm_u = np.linalg.norm(u_arr)
    norm_i = np.linalg.norm(i_arr)
    if norm_u == 0 or norm_i == 0:  # Avoid division by zero
        return 0.0
    return float(np.dot(u_arr, i_arr) / (norm_u * norm_i))

def register_cosine_similarity_udf(spark):
    """
    Register the cosine similarity UDF and return it.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.

    Returns
    -------
    UserDefinedFunction
        The registered cosine similarity UDF.
    """
    return F.udf(cosine_similarity_udf, "float")

def average_vectors(vectors, count):
    """
    Average a list of vectors.

    Parameters
    ----------
    vectors : list of Vectors
        The list of vectors to average.
    count : int
        The number of vectors.

    Returns
    -------
    Vector
        The averaged vector.
    """
    if count == 0 or not vectors:  # Avoid division by zero or empty vectors
        return Vectors.dense([0.0] * len(vectors[0].toArray())) if vectors else Vectors.dense([0.0])
    summed_vector = [sum(component) for component in zip(*[v.toArray() for v in vectors])]
    averaged_vector = [component / count for component in summed_vector]
    return Vectors.dense(averaged_vector)

def register_average_vector_udf(spark):
    """
    Register the average vector UDF and return it.

    Parameters
    ----------
    spark : SparkSession
        The Spark session.

    Returns
    -------
    UserDefinedFunction
        The registered average vector UDF.
    """
    return F.udf(average_vectors, VectorUDT())

def build_category_user_profiles(clicked_news_df, news_features_df):
    """
    Build category-specific user profiles by averaging the TF-IDF vectors of clicked articles per category.

    Parameters
    ----------
    clicked_news_df : DataFrame
        A DataFrame containing clicked news (with TFIDFeatures).
    news_features_df : DataFrame
        A DataFrame containing news features (e.g., TFIDFeatures, Category).

    Returns
    -------
    DataFrame
        A DataFrame with user profiles joined with news data for recommendations.
    """
    avg_vector_udf = register_average_vector_udf(clicked_news_df.sparkSession)

    category_user_profiles = clicked_news_df.groupBy("UserID", "Category").agg(
        collect_list("TFIDFeatures").alias("UserVectors"),
        size(collect_list("TFIDFeatures")).alias("CountVectors")
    ).withColumn(
        "UserProfile",
        avg_vector_udf("UserVectors", "CountVectors")
    ).drop("UserVectors", "CountVectors")

    user_recs_with_category = category_user_profiles.join(
        news_features_df,  # news DataFrame with category information
        on="Category",
        how="inner"
    )

    return user_recs_with_category
