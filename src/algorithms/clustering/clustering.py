import sys
import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# MongoDB Configuration
# ------------------------------------------------------------------------------
MONGO_URI = "mongodb://root:example@mongodb:27017/admin"  # Adjust as needed
DATABASE_NAME = "mind_news"
EMBEDDINGS_COLLECTION = "news_combined_embeddings"
CLUSTERED_NEWS_COLLECTION = "clustered_news_big_data"  # New collection for cluster assignments

# ------------------------------------------------------------------------------
# 1. Initialize Spark
# ------------------------------------------------------------------------------
def initialize_spark():
    """
    Initialize a SparkSession with all necessary configurations.
    """
    logger.info("Initializing SparkSession...")
    spark = (
        SparkSession.builder
        .appName("HierarchicalClusteringBigData")
        .master("local[*]")  # Adjust based on your cluster setup
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "1g")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config(
            "spark.jars.packages",
            "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1,"
            "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0"
        )
        .config("spark.mongodb.read.connection.uri", MONGO_URI)
        .config("spark.mongodb.write.connection.uri", MONGO_URI)
        .config("spark.mongodb.output.writeConcern.w", "1")
        .getOrCreate()
    )
    logger.info("SparkSession initialized.")
    return spark

# ------------------------------------------------------------------------------
# 2. UDF to Parse Embedding String into Array[Float]
# ------------------------------------------------------------------------------
@F.udf(returnType=ArrayType(FloatType()))
def to_float_array(embedding_str):
    """
    Parse a comma-separated string of floats into a Python list of floats.
    """
    if embedding_str is None:
        return []
    return [float(x) for x in embedding_str.split(",")]

# ------------------------------------------------------------------------------
# 3. Load & Parse Embeddings from MongoDB
# ------------------------------------------------------------------------------
def load_and_parse_embeddings(spark):
    """
    Load embeddings from MongoDB, parse the embedding strings, and return a DataFrame.
    """
    logger.info(f"Loading embeddings from MongoDB collection: {EMBEDDINGS_COLLECTION}")
    df = (
        spark.read.format("mongodb")
        .option("uri", MONGO_URI)
        .option("database", DATABASE_NAME)
        .option("collection", EMBEDDINGS_COLLECTION)
        .load()
    )

    # Convert the embedding_string to an array of floats
    df = df.withColumn("features", to_float_array(F.col("embedding_string")))

    # Select only the necessary fields
    df = df.select("_id", "news_id", "features")
    logger.info(f"Loaded {df.count()} documents from MongoDB.")
    return df

# ------------------------------------------------------------------------------
# 4. Convert to Spark Vectors & (Optional) Dimensionality Reduction
# ------------------------------------------------------------------------------
def prepare_features_for_clustering(df, pca_k=0):
    """
    Convert 'features' (array<float>) into Spark Dense Vector and apply PCA for dimensionality reduction.
    """
    logger.info("Converting array<float> to Spark Vector for ML...")

    # UDF to convert array<float> to Spark Vector
    @F.udf(returnType=VectorUDT())
    def array_to_vector(arr):
        return Vectors.dense(arr)

    df_vectors = df.withColumn("features_vector", array_to_vector(F.col("features")))

    # Apply PCA if pca_k is specified
    if pca_k and pca_k > 0:
        logger.info(f"Reducing dimensions to {pca_k} using PCA...")
        pca = PCA(k=pca_k, inputCol="features_vector", outputCol="pca_features")
        pca_model = pca.fit(df_vectors)
        df_vectors = pca_model.transform(df_vectors).select("_id", "news_id", F.col("pca_features").alias("final_features"))
        logger.info("PCA dimensionality reduction completed.")
    else:
        df_vectors = df_vectors.select("_id", "news_id", F.col("features_vector").alias("final_features"))

    return df_vectors

# ------------------------------------------------------------------------------
# 5. Stage 1 - K-Means Clustering (Spark) for Big Data
# ------------------------------------------------------------------------------
def kmeans_clustering(spark, df_vectors, k=500, max_iter=20):
    """
    Perform K-Means clustering using Spark ML on the dataset to generate k cluster centers.
    """
    logger.info(f"Running Spark K-Means with k={k}, max_iter={max_iter}...")
    kmeans = KMeans(
        k=k, 
        maxIter=max_iter, 
        featuresCol="final_features", 
        predictionCol="kmeans_cluster"
    )
    model = kmeans.fit(df_vectors)
    logger.info("K-Means clustering completed.")

    # Assign each document to one of the k clusters
    df_with_kmeans_labels = model.transform(df_vectors).select("_id", "news_id", "final_features", "kmeans_cluster")

    # Collect cluster centers to driver
    centers = model.clusterCenters()
    # Convert Spark Vectors to NumPy arrays
    centers_np = np.array([center.toArray() for center in centers])
    logger.info(f"K-Means cluster centers shape: {centers_np.shape}")

    return df_with_kmeans_labels, centers_np

# ------------------------------------------------------------------------------
# 6. Stage 2 - Hierarchical Clustering on K-Means Centers
# ------------------------------------------------------------------------------
def hierarchical_clustering_on_centers(centers_np, method='ward', threshold=5.0):
    """
    Perform hierarchical clustering on the K-Means cluster centers.
    """
    logger.info(f"Performing hierarchical clustering on {centers_np.shape[0]} centers using method='{method}' and threshold={threshold}...")
    linkage_matrix = linkage(centers_np, method=method)
    cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    logger.info("Hierarchical clustering on centers completed.")
    return cluster_labels, linkage_matrix

# ------------------------------------------------------------------------------
# 7. Stage 3 - Assign Each Data Point to Hierarchical Cluster
# ------------------------------------------------------------------------------
def map_points_to_hierarchical_clusters(spark, df_with_kmeans_labels, cluster_labels):
    """
    Map each news article to its hierarchical cluster based on K-Means cluster assignment.
    """
    logger.info("Mapping each K-Means cluster to its hierarchical cluster...")

    # Create a mapping DataFrame from K-Means cluster index to hierarchical cluster label
    kmeans_to_hier = [
        Row(kmeans_cluster=i, cluster=int(cluster_labels[i])) 
        for i in range(len(cluster_labels))
    ]
    lookup_df = spark.createDataFrame(kmeans_to_hier)

    # Join the hierarchical cluster labels with the original DataFrame
    df_final = (
        df_with_kmeans_labels.alias("df")
        .join(lookup_df.alias("lk"), F.col("df.kmeans_cluster") == F.col("lk.kmeans_cluster"), "inner")
        .select(
            F.col("df.news_id").alias("news_id"),
            F.col("lk.cluster").alias("cluster")
        )
    )
    logger.info("Mapping completed.")

    return df_final

# ------------------------------------------------------------------------------
# 8. Save Final Clusters to MongoDB
# ------------------------------------------------------------------------------
def save_clusters_to_mongodb(df_final):
    """
    Save the final hierarchical cluster assignments to MongoDB.
    Each document will contain 'news_id' and 'cluster'.
    """
    logger.info(f"Saving final cluster assignments to MongoDB collection: {CLUSTERED_NEWS_COLLECTION}...")

    (
        df_final.write.format("mongodb")
        .mode("overwrite")  
        .option("uri", MONGO_URI)
        .option("database", DATABASE_NAME)
        .option("collection", CLUSTERED_NEWS_COLLECTION)
        .save()
    )
    logger.info("Final cluster assignments saved successfully.")

# ------------------------------------------------------------------------------
# 9. (Optional) Evaluate & Visualize Clusters
# ------------------------------------------------------------------------------
def evaluate_clusters(centers_np, cluster_labels, linkage_matrix):
    """
    Evaluate the hierarchical clustering with Silhouette Score and save dendrogram plot.
    """
    from sklearn.metrics import silhouette_score

    logger.info("Evaluating hierarchical clustering of cluster centers...")
    if len(set(cluster_labels)) > 1:  # Silhouette Score requires at least 2 clusters
        sil_score = silhouette_score(centers_np, cluster_labels)
        logger.info(f"Silhouette Score (cluster centers): {sil_score}")
    else:
        logger.info("Only one cluster found - silhouette score not applicable.")

    # Plot dendrogram and save to file
    logger.info("Plotting dendrogram of cluster centers...")
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30)  # Adjust 'p' as needed
    plt.title("Hierarchical Clustering Dendrogram (K-Means Centers)")
    plt.xlabel("Cluster Index or (Cluster Size)")
    plt.ylabel("Distance")
    plt.savefig("dendrogram_centers.png")
    plt.close()
    logger.info("Dendrogram of cluster centers saved as 'dendrogram_centers.png'.")

# ------------------------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------------------------
def main():
    try:
        # 1. Initialize Spark
        spark = initialize_spark()

        # 2. Load & parse embeddings from MongoDB
        df = load_and_parse_embeddings(spark)

        # 3. Prepare features: vectorize and reduce dimensions (PCA)
        df_vectors = prepare_features_for_clustering(df, pca_k=50)

        # 4. Stage 1 - K-Means Clustering in Spark
        k = 500  # Adjust based on your dataset size and desired granularity
        df_kmeans, centers_np = kmeans_clustering(spark, df_vectors, k=k, max_iter=20)

        # 5. Stage 2 - Hierarchical Clustering on K-Means centers
        threshold = 5.0  # Adjust based on desired cluster granularity
        cluster_labels, linkage_matrix = hierarchical_clustering_on_centers(
            centers_np, method='ward', threshold=threshold
        )

        # 6. Stage 3 - Map each row to the final hierarchical cluster
        df_final = map_points_to_hierarchical_clusters(spark, df_kmeans, cluster_labels)

        # 7. Save final cluster assignments to MongoDB
        save_clusters_to_mongodb(df_final)

        # 8. (Optional) Evaluate & visualize the hierarchical clustering of centers
        evaluate_clusters(centers_np, cluster_labels, linkage_matrix)

        logger.info("Hierarchical clustering (big data) pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    finally:
        try:
            spark.stop()
            logger.info("SparkSession stopped.")
        except:
            pass

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
