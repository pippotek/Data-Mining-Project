import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import logging
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import faiss
import numpy as np
from sklearn.metrics import mean_squared_error

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_recommendations(
    mongo_uri,
    db_name,
    output_collection,
    user_ids,
    news_ids,
    similarity_scores,
    ranks
):
    """
    Save content-based recommendation results to MongoDB as a single document per user.

    Each document will have the following structure:
    {
        "_id": ObjectId(...),
        "userId": <user_id>,
        "recommendations": [
            {
                "newsId": <news_id>,
                "rating": <similarity_score>
            },
            ...
        ]
    }

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - output_collection (str): Collection name for saving recommendations.
    - user_ids (list): List of user IDs corresponding to the recommendations.
    - news_ids (list): List of news IDs recommended to the users.
    - similarity_scores (list): List of similarity scores for each recommendation.
    - ranks (list): List of ranks for each recommendation.

    Returns:
    - None
    """
    try:
        # Initialize MongoDB client
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[output_collection]

        # Group recommendations by user_id using a standard dictionary
        user_recs = {}
        for user_id, news_id, score, rank in zip(user_ids, news_ids, similarity_scores, ranks):
            if user_id not in user_recs:
                user_recs[user_id] = []
            user_recs[user_id].append({
                "newsId": news_id,
                "rating": float(score)  # Ensure compatibility
            })

        # Prepare bulk upsert operations
        operations = []
        for user_id, recs in user_recs.items():
            operations.append(
                UpdateOne(
                    {"userId": user_id},  # Filter by userId
                    {"$set": {"recommendations": recs}},  # Set recommendations array
                    upsert=True  # Insert if not exists
                )
            )

        if operations:
            logger.info(f"Preparing to insert/update {len(operations)} user documents into '{output_collection}' collection...")
            # Execute bulk operations
            result_bulk = collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk write completed: {result_bulk.bulk_api_result}")
        else:
            logger.info("No recommendations to save.")

    except BulkWriteError as bwe:
        logger.error(f"Bulk write error: {bwe.details}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Close MongoDB client
        client.close()


def create_recommendation_indexes(mongo_uri, db_name, collection_name):
    """
    Create necessary indexes on the recommendations collection.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - collection_name (str): Name of the recommendations collection.

    Returns:
    - None
    """
    try:
        # Initialize MongoDB client
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # Create unique index on 'userId' to prevent duplicate user documents
        collection.create_index(
            [("userId", 1)],
            unique=True,
            name="unique_userId_idx",
            background=True
        )
        logger.info("Created unique index on 'userId'.")

    except Exception as e:
        logger.error(f"Error creating indexes on '{collection_name}': {e}")
    finally:
        # Close MongoDB client
        client.close()


def load_data(mongo_uri, db_name, news_embeddings_collection, behaviors_train_collection, behaviors_test_collection):
    """
    Load data from MongoDB into Pandas DataFrames.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - news_embeddings_collection (str): Collection name for news embeddings.
    - behaviors_train_collection (str): Collection name for training behaviors.
    - behaviors_test_collection (str): Collection name for testing behaviors.

    Returns:
    - Tuple of DataFrames: (news_embeddings_df, behaviors_train_df, behaviors_test_df)
    """
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]

        logger.info(f"Loading collection: {news_embeddings_collection}")
        news_embeddings = pd.DataFrame(list(db[news_embeddings_collection].find()))

        logger.info(f"Loading collection: {behaviors_train_collection}")
        behaviors_train = pd.DataFrame(list(db[behaviors_train_collection].find()))

        logger.info(f"Loading collection: {behaviors_test_collection}")
        behaviors_test = pd.DataFrame(list(db[behaviors_test_collection].find()))

        client.close()
        return news_embeddings, behaviors_train, behaviors_test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_news_embeddings(news_embeddings_df):
    """
    Convert embedding_string column to a list of floats.

    Parameters:
    - news_embeddings_df (DataFrame): DataFrame containing news embeddings.

    Returns:
    - DataFrame: Updated DataFrame with numerical embeddings.
    """
    try:
        logger.info("Starting preprocessing of news embeddings...")
        
        def parse_embedding(s):
            try:
                # Split the string by commas and convert each element to float
                return np.array([float(x) for x in s.split(',')], dtype='float32')
            except Exception as e:
                logger.error(f"Failed to parse embedding string: {s[:50]}... | Error: {e}")
                return np.array([])  # Return an empty array for malformed embeddings

        # Apply the parsing function to the 'embedding_string' column
        news_embeddings_df['embedding'] = news_embeddings_df['embedding_string'].apply(parse_embedding)

        # Identify and remove rows with failed parsing (empty embeddings)
        initial_count = len(news_embeddings_df)
        news_embeddings_df = news_embeddings_df[news_embeddings_df['embedding'].map(len) > 0]
        removed_count = initial_count - len(news_embeddings_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} entries due to parsing errors.")

        # Drop the original 'embedding_string' column as it's no longer needed
        news_embeddings_df = news_embeddings_df.drop(columns=['embedding_string'])

        # Verify that all embeddings have the same dimension
        embedding_dimensions = news_embeddings_df['embedding'].apply(len).unique()
        if len(embedding_dimensions) != 1:
            logger.error(f"Inconsistent embedding dimensions found: {embedding_dimensions}")
            raise ValueError("All embeddings must have the same dimension.")
        else:
            logger.info(f"All embeddings have consistent dimension: {embedding_dimensions[0]}")

        return news_embeddings_df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


def average_embeddings(embeddings_list):
    """
    Compute the average of a list of embeddings.

    Parameters:
    - embeddings_list (list): List of NumPy arrays representing embeddings.

    Returns:
    - list: Averaged embedding as a list.
    """
    embeddings = np.array(embeddings_list)
    if len(embeddings) == 0:
        return []
    return np.mean(embeddings, axis=0).tolist()


def create_user_profiles(behaviors_train_df, news_embeddings_df):
    """
    Create user profiles by averaging embeddings for user history.

    Parameters:
    - behaviors_train_df (DataFrame): DataFrame containing training behaviors.
    - news_embeddings_df (DataFrame): DataFrame containing news embeddings.

    Returns:
    - DataFrame: DataFrame containing user profiles.
    """
    try:
        logger.info("Creating user profiles...")
        user_profiles = []

        # Precompute a dictionary for faster lookups
        news_embeddings_dict = news_embeddings_df.set_index('news_id')['embedding'].to_dict()

        # Ensure the 'history' column is a string and handle missing or invalid values
        behaviors_train_df['history'] = behaviors_train_df['history'].fillna("").astype(str)

        # Process each user
        for user_id, group in behaviors_train_df.groupby('user_id'):
            history_items = group['history'].iloc[0].split()  # Split the space-separated history into a list

            # Fetch embeddings using the precomputed dictionary
            embeddings = [
                news_embeddings_dict[news_id]
                for news_id in history_items if news_id in news_embeddings_dict
            ]

            if embeddings:
                user_profiles.append({
                    'user_id': user_id,
                    'user_embedding': average_embeddings(embeddings)
                })
            else:
                logger.warning(f"No valid embeddings found for user_id: {user_id}")

        user_profiles_df = pd.DataFrame(user_profiles)
        logger.info(f"Created profiles for {len(user_profiles_df)} users.")
        return user_profiles_df
    except Exception as e:
        logger.error(f"Error creating user profiles: {e}")
        raise



def build_faiss_index(news_embeddings_df, index_path='faiss_index.index', nlist=100):
    """
    Build and save a FAISS index for the news embeddings.

    Parameters:
    - news_embeddings_df (DataFrame): DataFrame containing news embeddings.
    - index_path (str): Path to save the FAISS index.
    - nlist (int): Number of clusters for the IVF index.

    Returns:
    - faiss.Index: Trained FAISS index.
    """
    try:
        logger.info("Building FAISS index...")

        # Convert embeddings to a numpy array
        embeddings = np.vstack(news_embeddings_df['embedding']).astype('float32')
        logger.info(f"Embeddings shape before normalization: {embeddings.shape}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        logger.info("Embeddings normalized.")

        d = embeddings.shape[1]  # Dimension of embeddings

        # Choose index type based on data size and desired speed/accuracy trade-off
        quantizer = faiss.IndexFlatIP(d)  # Inner product quantizer
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train the index
        logger.info("Training FAISS index...")
        index.train(embeddings)
        logger.info("FAISS index trained.")

        # Add embeddings to the index
        logger.info("Adding embeddings to FAISS index...")
        index.add(embeddings)
        logger.info("Embeddings added to FAISS index.")

        # Save the index to disk
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index built and saved to {index_path}")

        return index
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise





def load_faiss_index(index_path='faiss_index.index'):
    """
    Load a pre-built FAISS index from disk.

    Parameters:
    - index_path (str): Path to the FAISS index file.

    Returns:
    - faiss.Index: Loaded FAISS index.
    """
    try:
        logger.info(f"Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        logger.info("FAISS index loaded successfully.")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

def compute_recommendations_faiss_incremental(
    user_profiles_df,
    news_embeddings_df,
    index,
    top_k=10,
    chunk_size=1000
):
    """
    Compute recommendations using FAISS and return a DataFrame of recommendations.

    Parameters:
    - user_profiles_df: DataFrame containing user profiles.
    - news_embeddings_df: DataFrame containing news embeddings.
    - index: FAISS index object.
    - top_k: Number of top recommendations per user.
    - chunk_size: Number of users to process per chunk.

    Returns:
    - recommendations_df: DataFrame containing recommendations with columns ['user_id', 'news_id', 'similarity_score', 'rank']
    """
    try:
        logger.info("Computing recommendations using FAISS...")
        recommendations = []

        num_users = len(user_profiles_df)
        logger.info(f"Total number of users to process: {num_users}")

        for start in range(0, num_users, chunk_size):
            end = min(start + chunk_size, num_users)
            user_chunk = user_profiles_df.iloc[start:end]

            # Convert user embeddings to numpy array
            user_embeddings = np.vstack(user_chunk['user_embedding']).astype('float32')
            faiss.normalize_L2(user_embeddings)

            # Perform search
            distances, indices = index.search(user_embeddings, top_k)

            # Collect recommendations for the chunk
            for i, user_id in enumerate(user_chunk['user_id']):
                for rank, (dist, idx) in enumerate(zip(distances[i], indices[i]), start=1):
                    if idx == -1:
                        # No more neighbors
                        continue
                    news_id = news_embeddings_df.iloc[idx]['news_id']
                    similarity_score = float(dist)  # Convert to native float
                    recommendations.append({
                        'user_id': user_id,
                        'news_id': news_id,
                        'similarity_score': similarity_score,
                        'rank': rank
                    })

            logger.info(f"Processed users {start} to {end} out of {num_users}")

        recommendations_df = pd.DataFrame(recommendations)
        logger.info("Completed FAISS-based recommendations.")
        return recommendations_df
    except Exception as e:
        logger.error(f"Error during recommendation computation: {e}")
        raise


def load_recommendations(mongo_uri, db_name, recommendations_collection):
    """
    Load recommendations from MongoDB.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - recommendations_collection (str): Collection name for recommendations.

    Returns:
    - dict: Dictionary mapping userId to list of recommended newsIds.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    recs_cursor = db[recommendations_collection].find()
    recs = {}
    for doc in recs_cursor:
        user_id = doc['userId']
        recommended_news = [item['newsId'] for item in doc['recommendations']]
        recs[user_id] = recommended_news
    client.close()
    return recs


def load_ground_truth_parsed(mongo_uri, db_name, behaviors_test_collection):
    """
    Load and parse ground truth interactions from MongoDB.

    Parameters:
    - mongo_uri (str): MongoDB connection URI.
    - db_name (str): MongoDB database name.
    - behaviors_test_collection (str): Collection name for test behaviors.

    Returns:
    - dict: Dictionary mapping userId to set of relevant newsIds.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    gt_cursor = db[behaviors_test_collection].find()
    ground_truth = {}
    for doc in gt_cursor:
        user_id = doc['user_id']
        impressions = doc.get('impressions', '')
        # Split impressions and extract newsIds with click=1
        relevant_news = [item.split('-')[0] for item in impressions.split() if item.endswith('-1')]
        if user_id not in ground_truth:
            ground_truth[user_id] = set()
        ground_truth[user_id].update(relevant_news)
    client.close()
    return ground_truth

def calculate_mse_rmse(recommendations_df, ground_truth, k=10):
    """
    Calculate MSE and RMSE for the recommendations.
    
    Parameters:
    - recommendations_df (DataFrame): DataFrame with columns ['user_id', 'news_id', 'similarity_score', 'rank']
    - ground_truth (dict): Dictionary mapping user_id to set of relevant news_ids
    - k (int): Number of top recommendations per user
    
    Returns:
    - mse: Mean Squared Error
    - rmse: Root Mean Squared Error
    """
    labels = []
    predictions = []
    
    # Iterate through each recommendation
    for _, row in recommendations_df.iterrows():
        user_id = row['user_id']
        news_id = row['news_id']
        sim_score = row['similarity_score']
        
        # Assign label: 1 if news_id is relevant, else 0
        label = 1 if news_id in ground_truth.get(user_id, set()) else 0
        labels.append(label)
        predictions.append(sim_score)
    
    # Calculate MSE and RMSE
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    return mse, rmse

