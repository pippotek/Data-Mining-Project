import sys
import logging
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
# from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row
from sklearn.decomposition import PCA
import pymongo
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import os
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

def load_data(mongo_uri, db_name, news_embeddings_collection, news_collections):
    """
    Load data from MongoDB collections and remove duplicates based on 'news_id'.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - news_embeddings_collection (str): Collection name for news embeddings.
    - news_collections (list): List of collection names for news articles (e.g., ["news_train", "news_valid"]).

    Returns:
    - tuple: Lists of documents for news embeddings and deduplicated news articles.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    
    print(f"Fetching '{news_embeddings_collection}' collection...")
    news_embeddings = list(db[news_embeddings_collection].find())
    print(f"Fetched {len(news_embeddings)} documents from '{news_embeddings_collection}'.")

    # Initialize an empty list to hold all news articles from specified collections
    all_news = []
    for collection_name in news_collections:
        print(f"Fetching '{collection_name}' collection...")
        collection_data = list(db[collection_name].find())
        print(f"Fetched {len(collection_data)} documents from '{collection_name}'.")
        all_news.extend(collection_data)
    
    print(f"Total news articles loaded (including duplicates): {len(all_news)}")

    # Deduplicate articles based on 'news_id'
    deduplicated_news = {}
    for doc in all_news:
        news_id = doc.get('news_id')
        if news_id:
            # Update the dictionary with the latest document for the given news_id
            deduplicated_news[news_id] = doc
    
    deduplicated_list = list(deduplicated_news.values())
    print(f"Total news articles after deduplication: {len(deduplicated_list)}")
    
    return news_embeddings, deduplicated_list




def create_news_id_to_category_map(news):
    """
    Create a mapping from news_id to category.

    Parameters:
    - news (list): List of news documents.

    Returns:
    - dict: Mapping from news_id to category.
    """
    news_id_to_category = {}
    for doc in news:
        news_id = doc.get('news_id')
        category = doc.get('category', 'unknown')  # Default to 'unknown' if missing
        if news_id:
            news_id_to_category[news_id] = category
        else:
            print(f"Warning: Document with _id {doc.get('_id')} is missing 'news_id'.")
    return news_id_to_category


def parse_embeddings(news_embeddings, news_id_to_category):
    """
    Parse the embedding strings into NumPy arrays, along with news_id and category.

    Parameters:
    - news_embeddings (list): List of MongoDB documents with embedding strings.
    - news_id_to_category (dict): Mapping from news_id to category.

    Returns:
    - tuple: (NumPy array of embeddings, list of document IDs, list of news IDs, list of categories)
    """
    embeddings = []
    doc_ids = []
    news_ids = []
    categories = []
    
    for doc in news_embeddings:
        embedding_string = doc.get('embedding_string', '')
        if not embedding_string:
            print(f"Warning: Document with _id {doc.get('_id')} has empty 'embedding_string'. Skipping.")
            continue
        try:
            embedding = np.array([float(x) for x in embedding_string.split(',')])
        except ValueError as ve:
            print(f"Error parsing embedding for document _id {doc.get('_id')}: {ve}. Skipping.")
            continue
        embeddings.append(embedding)
        doc_ids.append(doc['_id'])
        news_id = doc.get('news_id', 'unknown')
        news_ids.append(news_id)
        category = news_id_to_category.get(news_id, 'unknown')
        categories.append(category)
    
    return np.array(embeddings), doc_ids, news_ids, categories


def perform_pca(embeddings, n_components):
    """
    Perform Principal Component Analysis (PCA) on the embeddings.

    Parameters:
    - embeddings (np.array): Array of embeddings.
    - n_components (int): Number of principal components.

    Returns:
    - np.array: PCA-transformed embeddings.
    - PCA: Fitted PCA model.
    """
    print(f"Performing PCA to reduce dimensionality to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"PCA completed. Reduced embeddings shape: {reduced_embeddings.shape}")
    return reduced_embeddings, pca


def perform_kmeans(reduced_embeddings, n_clusters):
    """
    Perform K-Means clustering on the reduced embeddings.

    Parameters:
    - reduced_embeddings (np.array): PCA-transformed embeddings.
    - n_clusters (int): Number of clusters.

    Returns:
    - np.array: Cluster labels for each data point.
    - KMeans: Fitted KMeans model.
    """
    print(f"Clustering data into {n_clusters} clusters using K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    print(f"K-Means clustering completed. Found clusters: {np.unique(cluster_labels)}")
    return cluster_labels, kmeans


def save_results(mongo_uri, db_name, output_collection, doc_ids, cluster_labels, news_ids, categories, pca_embeddings):
    """
    Save clustering results back to MongoDB with additional fields (news_id, category, cluster, and PCA embedding).

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - output_collection (str): Collection name for saving results.
    - doc_ids (list): List of document IDs corresponding to the embeddings.
    - cluster_labels (np.array): Cluster labels for each document.
    - news_ids (list): List of news IDs corresponding to the embeddings.
    - categories (list): List of categories for the embeddings.
    - pca_embeddings (np.array): PCA-transformed embeddings for each document.

    Returns:
    - None
    """
    from pymongo import MongoClient, UpdateOne
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    output_col = db[output_collection]

    results = []
    for doc_id, news_id, category, cluster, pca_embedding in zip(doc_ids, news_ids, categories, cluster_labels, pca_embeddings):
        result = {
            "_id": doc_id,
            "news_id": news_id,
            "category": category,
            "cluster": int(cluster),
            "pca_embedding": pca_embedding.tolist()  # Convert NumPy array to list for MongoDB compatibility
        }
        results.append(result)

    if results:
        print(f"Inserting/updating {len(results)} documents into '{output_collection}' collection...")
        # Bulk upsert operations for efficiency
        operations = [
            UpdateOne(
                {"_id": result["_id"]},
                {"$set": result},
                upsert=True
            )
            for result in results
        ]
        if operations:
            result_bulk = output_col.bulk_write(operations, ordered=False)
            print(f"Bulk write completed: {result_bulk.bulk_api_result}")
    else:
        print("No results to save.")

    

def visualize_clusters_tsne(reduced_embeddings, cluster_labels, save_path="src/outputs/clusters_visualization.png"):
    """
    Visualize clusters using t-SNE for dimensionality reduction and Matplotlib for plotting.

    Parameters:
    - reduced_embeddings (np.array): PCA-transformed embeddings.
    - cluster_labels (np.array): Cluster labels from K-Means.
    - save_path (str): Path to save the plot.

    Returns:
    - None
    """
    print("Reducing dimensions to 2D using t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(reduced_embeddings)
    print("t-SNE dimensionality reduction completed.")

    print("Plotting clusters...")
    plt.figure(figsize=(12, 10))
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=f"Cluster {cluster}",
            alpha=0.6,
            s=10  # Adjust point size as needed
        )
    plt.title("Cluster Visualization with t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Cluster visualization saved to '{save_path}'.")




def visualize_with_pca(reduced_embeddings, cluster_labels, save_path="src/outputs/clusters_visualization.png"):
    print("Using PCA for 2D visualization...")
    embeddings_2d = reduced_embeddings[:, :2]  # Use the first two PCA components

    print("Plotting clusters...")
    plt.figure(figsize=(10, 8))
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        indices = np.where(cluster_labels == cluster)
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=f"Cluster {cluster}",
            alpha=0.6
        )
    plt.title("Cluster Visualization (PCA)")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Cluster visualization saved to {save_path}.")





def analyze_category_distribution(mongo_uri, db_name, output_collection, csv_save_path, plot_save_path, heatmap_save_path):
    """
    Analyze and visualize the distribution of news categories within each cluster.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - output_collection (str): Collection name containing clustering results.
    - csv_save_path (str): Path to save the aggregated CSV file.
    - plot_save_path (str): Path to save the stacked bar chart.
    - heatmap_save_path (str): Path to save the heatmap.

    Returns:
    - None
    """
    from pymongo import MongoClient

    # -----------------------------
    # Connect to MongoDB
    # -----------------------------
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[output_collection]

    # -----------------------------
    # Fetch Clustering Results
    # -----------------------------
    print("Fetching clustering results from MongoDB...")
    cursor = collection.find({}, {"_id": 1, "news_id": 1, "category": 1, "cluster": 1})
    data = list(cursor)
    print(f"Fetched {len(data)} documents.")

    # -----------------------------
    # Load Data into DataFrame
    # -----------------------------
    df = pd.DataFrame(data)
    print("Data loaded into DataFrame.")

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    # Drop rows with missing 'category' or 'cluster'
    initial_shape = df.shape
    df = df.dropna(subset=['category', 'cluster'])
    cleaned_shape = df.shape
    print(f"Dropped {initial_shape[0] - cleaned_shape[0]} documents due to missing 'category' or 'cluster'.")

    # Ensure 'cluster' is integer
    df['cluster'] = df['cluster'].astype(int)

    # -----------------------------
    # Aggregate Counts
    # -----------------------------
    print("Aggregating category counts per cluster...")
    aggregation = df.groupby(['cluster', 'category']).size().reset_index(name='count')
    print("Aggregation completed.")

    # -----------------------------
    # Save Aggregated Data
    # -----------------------------
    print(f"Saving aggregated data to '{csv_save_path}'...")
    aggregation.to_csv(csv_save_path, index=False)
    print("Aggregated data saved.")

    # -----------------------------
    # Create Pivot Table for Visualization
    # -----------------------------
    print("Creating pivot table for visualization...")
    pivot_table = aggregation.pivot(index='cluster', columns='category', values='count').fillna(0).astype(int)
    print("Pivot table created.")

    # -----------------------------
    # Save Pivot Table as CSV (Optional)
    # -----------------------------
    pivot_table.to_csv("src/outputs/cluster_category_pivot_table.csv")
    print("Pivot table saved as 'cluster_category_pivot_table.csv'.")

    # -----------------------------
    # Visualization: Stacked Bar Chart
    # -----------------------------
    print("Generating stacked bar chart...")
    plt.figure(figsize=(12, 8))
    pivot_table.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab20', edgecolor='none')
    plt.title('Distribution of News Categories within Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of News Articles')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    plt.close()
    print(f"Stacked bar chart saved to '{plot_save_path}'.")

    # -----------------------------
    # Visualization: Heatmap
    # -----------------------------
    print("Generating heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    plt.title('Heatmap of News Category Distribution per Cluster')
    plt.xlabel('Category')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig(heatmap_save_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to '{heatmap_save_path}'.")





def main():
    """
    Main function to execute the PCA and clustering pipeline.
    """
    # -----------------------------
    # Configuration Parameters
    # -----------------------------
    MONGO_URI = "mongodb://root:example@mongodb:27017/admin"  # Adjust as needed
    DB_NAME = "mind_news"
    NEWS_EMBEDDINGS_COLLECTION = "news_combined_embeddings"
    NEWS_COLLECTIONS = ["news_train", "news_valid"]
    OUTPUT_COLLECTION = "news_combined_embeddings_processed"
    
    PCA_COMPONENTS = 50  # Number of principal components
    KMEANS_CLUSTERS = 3  # Number of clusters
    VISUALIZATION_PATH = "src/outputs/clusters_visualization_3.png"
    CATEGORY_DISTRIBUTION_CSV = "src/outputs/cluster_category_distribution.csv"
    CATEGORY_DISTRIBUTION_PLOT = "src/outputs/cluster_category_distribution.png"
    CATEGORY_DISTRIBUTION_HEATMAP = "src/outputs/cluster_category_distribution_heatmap.png"

    
    try:
        # -----------------------------
        # Load Data from MongoDB
        # -----------------------------
        print("Loading data from MongoDB...")
        
        news_embeddings, news = load_data(
            mongo_uri=MONGO_URI,
            db_name=DB_NAME,
            news_embeddings_collection=NEWS_EMBEDDINGS_COLLECTION,
            news_collections=NEWS_COLLECTIONS  # Pass the list of news collections
        )
    
        print(f"Loaded {len(news_embeddings)} news embeddings.")
        print(f"Loaded {len(news)} news articles.")
    
        # -----------------------------
        # Create news_id to category mapping
        # -----------------------------
        print("Creating news_id to category mapping...")
        news_id_to_category = create_news_id_to_category_map(news)
        print(f"Created mapping for {len(news_id_to_category)} news_ids.")
    
        # -----------------------------
        # Parse Embedding Strings and Assign Categories
        # -----------------------------
        print("Parsing embedding strings and assigning categories...")
        embeddings, doc_ids, news_ids, categories = parse_embeddings(news_embeddings, news_id_to_category)
        print(f"Parsed embeddings shape: {embeddings.shape}")
        print(f"Assigned categories to {len(categories)} embeddings.")
    
        # -----------------------------
        # Perform PCA
        # -----------------------------
        reduced_embeddings, pca_model = perform_pca(embeddings, n_components=PCA_COMPONENTS)
    
        # -----------------------------
        # Perform K-Means Clustering
        # -----------------------------
        cluster_labels, kmeans_model = perform_kmeans(reduced_embeddings, n_clusters=KMEANS_CLUSTERS)
    
        # -----------------------------
        # Save Results Back to MongoDB
        # -----------------------------
        print(f"Saving clustering results back to MongoDB collection '{OUTPUT_COLLECTION}'...")
        save_results(
            mongo_uri=MONGO_URI,
            db_name=DB_NAME,
            output_collection=OUTPUT_COLLECTION,
            doc_ids=doc_ids,
            cluster_labels=cluster_labels,
            news_ids=news_ids,
            categories=categories,
            pca_embeddings=reduced_embeddings  # Pass the PCA-transformed embeddings here
        )
        print("Data successfully saved to MongoDB.")
    
        # -----------------------------
        # Visualize Clusters
        # -----------------------------
        print("Starting cluster visualization...")
        visualize_with_pca(
            reduced_embeddings=reduced_embeddings,
            cluster_labels=cluster_labels,
            save_path=VISUALIZATION_PATH
        )
        print("Visualization completed and saved.")
        
        analyze_category_distribution(
            mongo_uri=MONGO_URI,
            db_name=DB_NAME,
            output_collection=OUTPUT_COLLECTION,
            csv_save_path=CATEGORY_DISTRIBUTION_CSV,
            plot_save_path=CATEGORY_DISTRIBUTION_PLOT,
            heatmap_save_path=CATEGORY_DISTRIBUTION_HEATMAP
        )
        print("Category distribution analysis completed.")

    
    except Exception as e:
        print("An error occurred:", e)
        raise


if __name__ == "__main__":
    main()
