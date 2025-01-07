from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf, col, explode, split, collect_list, expr
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import logging



# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark Session with MongoDB Connector

# MongoDB Configuration
MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
DATABASE_NAME = "mind_news"
news_embeddings_collection = "news_combined_embeddings"
behaviors_train_collection = "behaviors_train"
behaviors_test_collection = "behaviors_valid"
RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"

    # Initialize Spark Session with Optimized Configurations
spark = (SparkSession.builder
                .appName("Combine News and Generate Embeddings")
                .master("local[*]")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "2g") \
                .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "128m")
                .config("spark.jars.packages",
                        "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
                        "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
                .config("spark.mongodb.read.connection.uri", MONGO_URI)
                .config("spark.mongodb.write.connection.uri", MONGO_URI)
                .getOrCreate())

logger.info("Spark Session initialized.")

def fetch_data_from_mongo(spark: SparkSession, uri: str, db_name: str, collection_name: str):
    """
    Fetch data from a MongoDB collection into a PySpark DataFrame.

    Parameters
    ----------
    spark : SparkSession
        A SparkSession object, already configured to use the Mongo Spark connector.
    uri : str
        The MongoDB connection URI (e.g. "mongodb://user:password@mongodb:27017").
    db_name : str
        The name of the MongoDB database.
    collection_name : str
        The name of the collection to read from.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark DataFrame containing the data from the specified MongoDB collection.
    """
    df = (spark.read
               .format("mongodb")
               .option("uri", uri)
               .option("database", db_name)
               .option("collection", collection_name)
               .load())
    return df

def load_data(spark, mongo_uri, db_name, news_embeddings_collection, behaviors_train_collection, behaviors_test_collection):
    """
    Load data from MongoDB into Spark DataFrames.
    """
    news_embeddings_df = fetch_data_from_mongo(spark, mongo_uri, db_name, news_embeddings_collection)
    behaviors_train_df = fetch_data_from_mongo(spark, mongo_uri, db_name, behaviors_train_collection)
    behaviors_test_df = fetch_data_from_mongo(spark, mongo_uri, db_name, behaviors_test_collection)
    return news_embeddings_df, behaviors_train_df, behaviors_test_df

def preprocess_news_embeddings(news_embeddings_df):
    """
    Convert embedding_string column to an array of floats and optimize data types.
    """
    logger.info("Preprocessing news embeddings...")
    
    # Define a UDF to parse the embedding string into an array of floats
    def parse_embedding(embedding_str):
        return [float(x) for x in embedding_str.split(',')]
    
    parse_embedding_udf = udf(parse_embedding, ArrayType(FloatType()))
    
    # Apply the UDF and select only necessary columns
    news_embeddings_df = news_embeddings_df.withColumn("embedding", parse_embedding_udf(col("embedding_string"))) \
                                           .drop("embedding_string") \
                                           .select("news_id", "embedding")
    
    logger.info("Preprocessed news embeddings.")
    return news_embeddings_df
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

def convert_array_to_vector(df, array_col, vector_col):
    """
    Convert an array column to a Vector column.
    """
    logger.info(f"Converting {array_col} from array<float> to vector<float>...")
    vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
    df = df.withColumn(vector_col, vector_udf(col(array_col)))
    return df

def apply_pca(df, input_col, output_col, pca_components=128):
    """
    Apply PCA to reduce the dimensionality of embeddings.
    
    Parameters:
    - df: Spark DataFrame containing the input column.
    - input_col: Name of the Vector column to apply PCA on.
    - output_col: Name of the output column with reduced dimensions.
    - pca_components: Number of principal components.
    
    Returns:
    - pca_model: Trained PCA model.
    - df_pca: DataFrame with the reduced-dimensionality embeddings.
    """
    logger.info(f"Applying PCA to reduce dimensionality to {pca_components} components...")
    pca = PCA(k=pca_components, inputCol=input_col, outputCol=output_col)
    pca_model = pca.fit(df)
    df_pca = pca_model.transform(df).select("news_id", output_col)
    logger.info("PCA transformation completed.")
    return pca_model, df_pca


def build_faiss_index(news_embeddings_pca_df):
    """
    Build a FAISS index from PCA-reduced news embeddings.
    
    Parameters:
    - news_embeddings_pca_df: Spark DataFrame with 'news_id' and 'embedding_pca'.
    
    Returns:
    - index: FAISS index.
    - news_ids: List of news IDs corresponding to the embeddings.
    """
    logger.info("Building FAISS index with PCA-reduced embeddings...")
    
    # Convert Spark DataFrame to Pandas DataFrame
    news_embeddings_pd = news_embeddings_pca_df.toPandas()
    logger.info("Converted Spark DataFrame to Pandas DataFrame.")
    
    # Extract embeddings and news_ids
    embeddings = np.vstack(news_embeddings_pd['embedding_pca'].values).astype('float32')
    news_ids = news_embeddings_pd['news_id'].tolist()
    logger.info(f"Extracted {len(news_ids)} news embeddings for FAISS indexing.")
    
    # Normalize embeddings for cosine similarity
    embeddings_normalized = normalize(embeddings, axis=1)
    
    # Build FAISS index
    d = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
    index.add(embeddings_normalized)
    logger.info(f"FAISS index built with {index.ntotal} vectors.")
    
    return index, news_ids

def create_user_profiles(behaviors_train_df, news_embeddings_pca_df, pca_model):
    """
    Create user profiles by averaging PCA-transformed embeddings of their history.
    
    Parameters:
    - behaviors_train_df: Spark DataFrame with user behaviors.
    - news_embeddings_pca_df: Spark DataFrame with 'news_id' and 'embedding_pca'.
    - pca_model: Trained PCA model.
    
    Returns:
    - user_profiles_pca_df: Spark DataFrame with 'user_id' and 'user_embedding_pca' (array<float>).
    """
    logger.info("Creating user profiles in PCA space...")
    
    # Explode the history field assuming it's a space-separated string
    behaviors_train_df = behaviors_train_df.withColumn("history_item", explode(split(col("history"), " ")))
    
    # Join with PCA-transformed news embeddings to get embeddings for each history item
    joined_df = behaviors_train_df.join(news_embeddings_pca_df, behaviors_train_df.history_item == news_embeddings_pca_df.news_id, "left") \
                                   .select("user_id", "embedding_pca")
    
    # Group by user_id and collect embeddings into a list
    user_profiles_pca_df = joined_df.groupBy("user_id") \
                                    .agg(collect_list("embedding_pca").alias("embeddings_pca")) \
                                    .withColumn("user_embedding_pca", average_embeddings_udf(col("embeddings_pca"))) \
                                    .select("user_id", "user_embedding_pca")
    
    logger.info(f"Created profiles for {user_profiles_pca_df.count()} users in PCA space.")
    return user_profiles_pca_df

# Define a UDF to compute the average of embeddings
def average_embeddings(embeddings):
    """
    Compute the average of a list of embeddings.
    
    Parameters:
    - embeddings: List of lists (embeddings).
    
    Returns:
    - average_embedding: List representing the average embedding.
    """
    embeddings_np = np.array(embeddings)
    if embeddings_np.size == 0:
        return []
    return np.mean(embeddings_np, axis=0).tolist()

# Register the UDF with Spark
average_embeddings_udf = udf(average_embeddings, ArrayType(FloatType()))

from pyspark.sql.functions import monotonically_increasing_id, floor

def add_distributed_index(df, batch_size):
    """
    Adds a unique index to each row in a distributed manner.
    """
    df_with_id = df.withColumn("unique_id", monotonically_increasing_id())
    # Assign batch number based on unique_id
    df_with_batch = df_with_id.withColumn("batch_num", floor(F.col("unique_id") / batch_size))
    return df_with_batch

def compute_recommendations(user_profiles_df, faiss_index, news_ids, top_k=10, batch_size=50):
    """
    Compute recommendations for each user using the FAISS index in batches.
    
    Parameters:
    - user_profiles_df: Spark DataFrame with 'user_id' and 'user_embedding_pca'.
    - faiss_index: FAISS index built from news embeddings.
    - news_ids: List of news IDs corresponding to the FAISS index.
    - top_k: Number of top recommendations to retrieve.
    - batch_size: Number of users to process in each batch.
    
    Returns:
    - recommendations_df: Spark DataFrame with recommendations for each user.
    """
    import math
    
    logger.info("Starting batch processing for recommendations...")
    
    # Add distributed index
    user_profiles_with_batch = add_distributed_index(user_profiles_df, batch_size)
    
    # Calculate total users and number of batches
    total_users = user_profiles_with_batch.count()
    num_batches = math.ceil(total_users / batch_size)
    logger.info(f"Total users: {total_users}, Batch size: {batch_size}, Number of batches: {num_batches}")
    
    # Initialize list to store recommendations
    recommendations = []
    
    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size
        
        logger.info(f"Processing batch {batch_num + 1}/{num_batches}: Users {start} to {end - 1}")
        
        # Filter the batch
        batch_df = user_profiles_with_batch.filter(
            (F.col("batch_num") == batch_num)
        ).drop("unique_id", "batch_num", "index")  # Drop unnecessary columns
        
        user_profiles_pd = batch_df.toPandas()

        # Check if the batch is empty
        if user_profiles_pd.empty:
            logger.info(f"Batch {batch_num + 1} is empty. Skipping.")
            continue
        
        # Extract user embeddings
        user_profiles_pd = user_profiles_pd.dropna(subset = 'user_embedding_pca')
        user_embeddings = np.vstack(user_profiles_pd['user_embedding_pca'].values).astype('float32')
        
        # Normalize embeddings for cosine similarity
        user_embeddings_normalized = normalize(user_embeddings, axis=1)
        
        # Perform FAISS search
        distances, indices = faiss_index.search(user_embeddings_normalized, top_k)
        
        # Prepare recommendations
        for user, dist, idx in zip(user_profiles_pd['user_id'], distances, indices):
            rec_news_ids = [news_ids[j] for j in idx]
            rec_scores = dist.tolist()
            rec_ranks = list(range(1, len(rec_news_ids) + 1))
            recommendations.append({
                "user_id": user,
                "news_id": rec_news_ids,
                "similarity_score": rec_scores,
                "rank": rec_ranks
            })
        
        logger.info(f"Completed batch {batch_num + 1}/{num_batches}")
    
    # Convert recommendations to Spark DataFrame
    recommendations_pd = pd.DataFrame(recommendations)
    recommendations_df = spark.createDataFrame(recommendations_pd)
    logger.info("All recommendations have been computed and converted to Spark DataFrame.")
    
    return recommendations_df




def save_recommendations_to_mongodb(recommendations_df, mongo_uri, db_name, output_collection):
    """
    Save recommendations to MongoDB as per the specified document structure.
    """
    logger.info("Saving recommendations to MongoDB...")
    
    # Assemble the recommendations array
    recommendations_final = recommendations_df.withColumn("recommendations", 
        expr("""
        transform(
            sequence(1, size(news_id)),
            x -> struct(
                news_id[x - 1] as newsId,
                similarity_score[x - 1] as rating,
                rank[x - 1] as rank
            )
        )
        """)
    ).select("user_id", "recommendations") \
     .withColumnRenamed("user_id", "userId")
    
    # Write to MongoDB
    recommendations_final.write.format("mongo") \
        .mode("append") \
        .option("uri", mongo_uri) \
        .option("database", db_name) \
        .option("collection", output_collection) \
        .save()
    
    logger.info(f"Saved recommendations to MongoDB collection: {output_collection}")

def main():
    """
    Main function to execute the recommendation pipeline with PCA integration.
    """
    # Configurations
    DATABASE_NAME = 'mind_news'
    news_embeddings_collection = "news_combined_embeddings"
    behaviors_train_collection = "behaviors_train"
    behaviors_test_collection = "behaviors_valid"
    RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"
    PCA_COMPONENTS = 128  # Adjust based on desired dimensionality reduction
    
    news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
        spark,
        MONGO_URI,
        DATABASE_NAME,
        news_embeddings_collection,
        behaviors_train_collection,
        behaviors_test_collection
    )
    
    # Preprocess embeddings
    news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)
    
    # Convert array<float> to Vector
    news_embeddings_df = convert_array_to_vector(news_embeddings_df, "embedding", "embedding_vector")
    
    # Apply PCA
    pca_model, news_embeddings_pca_df = apply_pca(
        news_embeddings_df,
        input_col="embedding_vector",
        output_col="embedding_pca",
        pca_components=PCA_COMPONENTS
    )
    
    # **Pass the pca_model to create_user_profiles**
    user_profiles_df = create_user_profiles(behaviors_train_df, news_embeddings_pca_df, pca_model)
    
    # Build FAISS index
    faiss_index, news_ids = build_faiss_index(news_embeddings_pca_df)
    
    
    # Compute recommendations using FAISS via Pandas UDF
    recommendations_df = compute_recommendations(
        user_profiles_df=user_profiles_df,
        faiss_index=faiss_index,
        news_ids=news_ids,
        top_k=10
    )

    recommendations_df.limit(10).show()

if __name__ == "__main__":
    main()
