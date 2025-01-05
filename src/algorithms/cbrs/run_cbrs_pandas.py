from src.algorithms.cbrs.cbrs_utils_pandas import * 
from src.utilities.data_utils import * 
from src.algorithms.cbrs.clean_embed import main_embedding
from cbrs_utils_pandas import *
import logging
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to execute the recommendation pipeline.
    """
    # Configurations
    MONGO_URI = "mongodb://root:example@mongodb:27017/admin"
    DATABASE_NAME = "mind_news"
    news_embeddings_collection = "news_combined_embeddings"
    behaviors_train_collection = "behaviors_train"
    behaviors_test_collection = "behaviors_valid"
    RECOMMENDATIONS_COLLECTION = "cbrs_recommendations"
    INDEX_PATH = '/app/src/algorithms/cbrs/index/faiss_index.index'
    TOP_K = 10

    # Ensure the index directory exists
    import os
    index_dir = os.path.dirname(INDEX_PATH)
    os.makedirs(index_dir, exist_ok=True)

    # Load data from MongoDB
    logger.info("Loading data from MongoDB...")
    news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
        mongo_uri=MONGO_URI,
        db_name=DATABASE_NAME,
        news_embeddings_collection=news_embeddings_collection,
        behaviors_train_collection=behaviors_train_collection,
        behaviors_test_collection=behaviors_test_collection
    )

    # Preprocess embeddings
    logger.info("Preprocessing news embeddings...")
    news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)

    # Verify embeddings
    if news_embeddings_df.empty:
        logger.error("No valid embeddings available after preprocessing.")
        return

    # Create user profiles
    logger.info("Creating user profiles...")
    user_profiles_df = create_user_profiles(behaviors_train_df, news_embeddings_df)

    # Verify user profiles
    if user_profiles_df.empty:
        logger.error("No user profiles created. Check training behaviors and embeddings.")
        return

    # Build or load FAISS index
    try:
        index = load_faiss_index(INDEX_PATH)
        logger.info("FAISS index loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load FAISS index: {e}. Building a new index...")
        index = build_faiss_index(news_embeddings_df, index_path=INDEX_PATH)

    # Compute recommendations using FAISS
    logger.info("Computing recommendations using FAISS...")
    recommendations_df = compute_recommendations_faiss_incremental(
        user_profiles_df=user_profiles_df,
        news_embeddings_df=news_embeddings_df,  # Ensure this is a DataFrame
        index=index,
        top_k=TOP_K,
        chunk_size=1000  # Adjust based on memory and performance
    )

    # Save recommendations
    logger.info("Saving recommendations to MongoDB...")
    save_recommendations(
        mongo_uri=MONGO_URI,
        db_name=DATABASE_NAME,
        output_collection=RECOMMENDATIONS_COLLECTION,
        user_ids=recommendations_df['user_id'].tolist(),
        news_ids=recommendations_df['news_id'].tolist(),
        similarity_scores=recommendations_df['similarity_score'].tolist(),
        ranks=recommendations_df['rank'].tolist()
    )

    # Load ground truth and recommendations
    logger.info("Loading ground truth data...")
    ground_truth = load_ground_truth_parsed(MONGO_URI, DATABASE_NAME, behaviors_test_collection)

    # Load saved recommendations
    logger.info("Loading saved recommendations...")
    saved_recommendations = load_recommendations(MONGO_URI, DATABASE_NAME, RECOMMENDATIONS_COLLECTION)

    # Check overlapping users
    overlapping_users = set(saved_recommendations.keys()) & set(ground_truth.keys())
    logger.info(f"Number of overlapping users: {len(overlapping_users)}")
    if len(overlapping_users) == 0:
        logger.error("No overlapping users between recommendations and ground truth.")
        return

    # Filter recommendations to only overlapping users
    filtered_recommendations_df = recommendations_df[recommendations_df['user_id'].isin(overlapping_users)]


    # Calculate MSE and RMSE
    logger.info("Calculating MSE and RMSE...")
    mse, rmse = calculate_mse_rmse(filtered_recommendations_df, ground_truth, k=TOP_K)
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    

if __name__ == "__main__":
    main()
