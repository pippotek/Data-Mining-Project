import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import logging
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import faiss

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


import logging
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

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
        
        existing_indexes = collection.index_information()
        if "unique_userId_idx" in existing_indexes:
            logger.info("Index already exists.")

    except Exception as e:
        logger.error(f"Error creating indexes on '{collection_name}': {e}")
    finally:
        # Close MongoDB client
        client.close()


def load_data(mongo_uri, db_name, news_embeddings_collection, behaviors_train_collection, behaviors_test_collection):
    """
    Load data from MongoDB into Pandas DataFrames.
    """
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

def preprocess_news_embeddings(news_embeddings_df):
    """
    Convert embedding_string column to a list of floats.
    """
    def safe_convert(s):
        try:
            return np.array([float(x) for x in s.split(',')])
        except Exception as e:
            logger.error(f"Error parsing embedding string: {s} - {e}")
            return None

    news_embeddings_df['embedding'] = news_embeddings_df['embedding_string'].apply(safe_convert)
    news_embeddings_df = news_embeddings_df.dropna(subset=['embedding'])
    
    return news_embeddings_df

def average_embeddings(embeddings_list):
    """
    Compute the average of a list of embeddings.
    """
    embeddings = np.array(embeddings_list)
    if len(embeddings) == 0:
        return np.zeros(embeddings[0].shape).tolist()
    
    return np.mean(embeddings, axis=0).tolist()

def create_user_profiles(behaviors_train_df, news_embeddings_df):
    """
    Create user profiles by averaging embeddings for user history.
    Optimized for faster lookups using a dictionary.
    """
    logger.info("Creating user profiles...")
    user_profiles = []

    # Precompute a dictionary for faster lookups
    news_embeddings_dict = news_embeddings_df.set_index('news_id')['embedding'].to_dict()

    # Ensure the 'history' column is a string and handle missing or invalid values
    behaviors_train_df['history'] = behaviors_train_df['history'].fillna("").astype(str)
    
    missing_ids = [news_id for news_id in history_items if news_id not in news_embeddings_dict]
    if missing_ids:
        logger.warning(f"Missing embeddings for news IDs: {missing_ids}")
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

    user_profiles_df = pd.DataFrame(user_profiles)
    logger.info(f"Created profiles for {len(user_profiles_df)} users.")
    return user_profiles_df


def create_user_profiles_weighted(behaviors_train_df, news_embeddings_df):
    """
    Create user profiles by averaging embeddings for user history with weighted averaging.
    More recent interactions are given higher weights.
    """
    logger.info("Creating user profiles with weighted averaging...")
    user_profiles = []

    # Precompute a dictionary for faster lookups
    news_embeddings_dict = news_embeddings_df.set_index('news_id')['embedding'].to_dict()

    # Ensure the 'history' column is a string and handle missing or invalid values
    behaviors_train_df['history'] = behaviors_train_df['history'].fillna("").astype(str)

    # Process each user
    for user_id, group in behaviors_train_df.groupby('user_id'):
        history_items = group['history'].iloc[0].split()  # Split the space-separated history into a list

        # Reverse history to give recent items higher weights
        weights = [1 / (i + 1) for i in range(len(history_items))]
        weights = np.array(weights) / sum(weights)  # Normalize weights to sum to 1

        # Fetch embeddings using the precomputed dictionary
        embeddings = [
            news_embeddings_dict[news_id]
            for news_id in history_items if news_id in news_embeddings_dict
        ]

        if embeddings:
            # Apply weights to embeddings
            embeddings = np.array(embeddings)
            weighted_embedding = np.average(embeddings, axis=0, weights=weights[:len(embeddings)])
            user_profiles.append({
                'user_id': user_id,
                'user_embedding': weighted_embedding.tolist()
            })

    user_profiles_df = pd.DataFrame(user_profiles)
    logger.info(f"Created profiles for {len(user_profiles_df)} users with weighted averaging.")
    return user_profiles_df


def build_faiss_index(news_embeddings_df, index_path='faiss_index.index', nlist=100):
    """
    Build and save a FAISS index for the news embeddings.
    """
    logger.info("Building FAISS index...")

    # Convert embeddings to a numpy array
    embeddings = np.vstack(news_embeddings_df['embedding']).astype('float32')
    embeddings_normalized = normalize(embeddings, axis=1)
    if embeddings.size == 0:
        logger.error("No embeddings found. Cannot build FAISS index.")
        return None
    if not np.isfinite(embeddings).all():
        logger.error("Embeddings contain NaN or infinite values.")
        return None
    
    d = embeddings_normalized.shape[1]  # Dimension of embeddings

    # Choose index type based on data size and desired speed/accuracy trade-off
    quantizer = faiss.IndexFlatIP(d)  # Inner product quantizer
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_normalized)

    # Train the index
    logger.info("Training FAISS index...")
    index.train(embeddings_normalized)

    # Add embeddings to the index
    logger.info("Adding embeddings to FAISS index...")
    index.add(embeddings_normalized)

    # Save the index to disk
    faiss.write_index(index, index_path)
    logger.info(f"FAISS index built and saved to {index_path}")

    return index

def load_faiss_index(index_path='faiss_index.index'):
    """
    Load a pre-built FAISS index from disk.
    """
    logger.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    return index

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
    logger.info("Computing recommendations using FAISS...")
    recommendations = []
    valid_indices = [idx for idx in indices[i] if idx < len(news_embeddings_df)]
    if not valid_indices:
        logger.warning(f"No valid indices for user {user_id}. Skipping recommendations.")
    
        num_users = len(user_profiles_df)

    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)
        user_chunk = user_profiles_df.iloc[start:end]

        user_embeddings = np.vstack(user_chunk['user_embedding']).astype('float32')
        faiss.normalize_L2(user_embeddings)

        # Perform search
        distances, indices = index.search(user_embeddings, top_k)

        # Collect recommendations for the chunk
        for i, user_id in enumerate(user_chunk['user_id']):
            for rank, (dist, idx) in enumerate(zip(distances[i], indices[i]), start=1):
                news_id = news_embeddings_df.iloc[idx]['news_id']
                similarity_score = float(dist)  # Convert to native float
                recommendations.append({
                    'user_id': user_id,
                    'news_id': news_id,
                    'similarity_score': similarity_score,
                    'rank': rank
                })

    recommendations_df = pd.DataFrame(recommendations)
    logger.info("Completed FAISS-based recommendations.")
    return recommendations_df



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
    try:
        relevant_news = [item.split('-')[0] for item in impressions.split() if item.endswith('-1')]
    except Exception as e:
        logger.error(f"Error parsing impressions for user {user_id}: {e}")

        if user_id not in ground_truth:
            ground_truth[user_id] = set()
        ground_truth[user_id].update(relevant_news)
    client.close()
    return ground_truth


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


def precision_at_k(recommended, relevant, k=10):
    """
    Compute Precision@K for a single user.
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    return len(recommended_set & relevant_set) / k

def recall_at_k(recommended, relevant, k=10):
    """
    Compute Recall@K for a single user.
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    if len(relevant_set) == 0:
        return 0
    return len(recommended_set & relevant_set) / len(relevant_set)

def ndcg_at_k(recommended, relevant, k=10):
    """
    Compute NDCG@K for a single user.
    """
    dcg = 0.0
    for i in range(k):
        if i >= len(recommended):
            break
        item = recommended[i]
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # Rank starts at 1
    # Compute ideal DCG
    ideal_relevant = sorted(relevant, key=lambda x: 1, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_relevant))):
        idcg += 1 / np.log2(i + 2)
    if idcg == 0:
        return 0
    return dcg / idcg

def average_precision(recommended, relevant, k=10):
    """
    Compute Average Precision@K for a single user.
    """
    ap = 0.0
    hit_count = 0
    for i in range(k):
        if i >= len(recommended):
            break
        if recommended[i] in relevant:
            hit_count += 1
            ap += hit_count / (i + 1)
    if hit_count == 0:
        return 0
    return ap / hit_count

def hit_rate_at_k(recommended, relevant, k=10):
    """
    Compute Hit Rate@K for a single user.
    """
    recommended_k = recommended[:k]
    relevant_set = set(relevant)
    recommended_set = set(recommended_k)
    return int(len(recommended_set & relevant_set) > 0)


def evaluate_recommendations(recommendations, ground_truth, k=10):
    """
    Evaluate recommendations using various metrics.
    
    Parameters:
    - recommendations (dict): userId -> list of recommended newsIds.
    - ground_truth (dict): userId -> set of relevant newsIds.
    - k (int): The cutoff rank.
    
    Returns:
    - dict: Dictionary containing average metrics.
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    average_precisions = []
    hit_rates = []
    
    for user_id, recommended in recommendations.items():
        relevant = ground_truth.get(user_id, set())
        if not relevant:
            continue  # Skip users with no ground truth
        
        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        ndcg = ndcg_at_k(recommended, relevant, k)
        ap = average_precision(recommended, relevant, k)
        hit = hit_rate_at_k(recommended, relevant, k)
        
        precision_scores.append(p)
        recall_scores.append(r)
        ndcg_scores.append(ndcg)
        average_precisions.append(ap)
        hit_rates.append(hit)
    
    # Compute average metrics
    metrics = {
        'Precision@{}'.format(k): np.mean(precision_scores) if precision_scores else 0,
        'Recall@{}'.format(k): np.mean(recall_scores) if recall_scores else 0,
        'NDCG@{}'.format(k): np.mean(ndcg_scores) if ndcg_scores else 0,
        'MAP@{}'.format(k): np.mean(average_precisions) if average_precisions else 0,
        'Hit Rate@{}'.format(k): np.mean(hit_rates) if hit_rates else 0
    }
    
    return metrics




# def debug_and_insights(news_embeddings_df, behaviors_train_df, ground_truth, recommendations):
#     """
#     Perform debugging and gather insights about the recommendation system.
#     """
#     relevant_counts = [len(items) for items in ground_truth.values()]
#     logger.info(f"Average relevant items per user: {np.mean(relevant_counts):.2f}")
#     logger.info("Sample ground truth data:")
#     logger.info(list(ground_truth.items())[:5])

#     logger.info("Sample recommendations data:")
#     logger.info(list(recommendations.items())[:5])

#     coverage = len(recommendations) / len(ground_truth)
#     logger.info(f"Recommendation coverage: {coverage:.4f}")

#     embedding_dims = [len(embedding) for embedding in news_embeddings_df['embedding']]
#     logger.info(f"Average embedding dimension: {np.mean(embedding_dims):.2f}")
#     logger.info("Sample news embeddings:")
#     logger.info(news_embeddings_df['embedding'].iloc[:5])

#     logger.info("Sample user profiles:")
#     user_profiles_df = create_user_profiles(behaviors_train_df, news_embeddings_df)
#     logger.info(user_profiles_df.head())

#     missing_profiles = set(ground_truth.keys()) - set(user_profiles_df['user_id'])
#     logger.info(f"Users without profiles: {len(missing_profiles)}")

#     history_lengths = behaviors_train_df['history'].apply(lambda x: len(x.split()))
#     logger.info(f"Average history length: {np.mean(history_lengths):.2f}")

#     invalid_histories = behaviors_train_df['history'].apply(
#         lambda x: [item for item in x.split() if item not in news_embeddings_df['news_id'].values]
#     )
#     logger.info(f"Invalid history items: {len(invalid_histories)}")

#     for user_id in list(ground_truth.keys())[:5]:
#         relevant_items = ground_truth[user_id]
#         recommended_items = [rec['newsId'] for rec in recommendations.get(user_id, [])]
#         logger.info(f"User {user_id} - Relevant: {relevant_items}, Recommended: {recommended_items}")

#     # Baseline analysis
#     popular_items = news_embeddings_df['news_id'].value_counts().index[:10]
#     baseline_recommendations = {user: popular_items for user in ground_truth.keys()}
#     baseline_metrics = evaluate_recommendations(baseline_recommendations, ground_truth, k=10)
#     logger.info(f"Baseline Metrics: {baseline_metrics}")

#     # Example metrics for a single user
#     user_id = list(ground_truth.keys())[0]
#     recommended = [rec['newsId'] for rec in recommendations[user_id]]
#     relevant = ground_truth[user_id]

#     precision = precision_at_k(recommended, relevant, k=10)
#     recall = recall_at_k(recommended, relevant, k=10)
#     ndcg = ndcg_at_k(recommended, relevant, k=10)
#     logger.info(f"User {user_id} - Precision: {precision}, Recall: {recall}, NDCG: {ndcg}")

#     unique_recommended_items = set(item['newsId'] for recs in recommendations.values() for item in recs)
#     logger.info(f"Unique recommended items: {len(unique_recommended_items)}")

#     total_relevant_items = set(item for items in ground_truth.values() for item in items)
#     recommended_items = set(item['newsId'] for recs in recommendations.values() for item in recs)
#     coverage = len(recommended_items & total_relevant_items) / len(total_relevant_items)
#     logger.info(f"Recommendation coverage: {coverage:.4f}")
