from src.algorithms.cbrs.cbrs_utils_pandas import * 
from src.utilities.data_utils import * 
from src.algorithms.cbrs.clean_embed import main_embedding
from cbrs_utils import *
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

    # Ensure these are Pandas DataFrames loaded from MongoDB
    news_embeddings_df, behaviors_train_df, behaviors_test_df = load_data(
        mongo_uri=MONGO_URI,
        db_name=DATABASE_NAME,
        news_embeddings_collection=news_embeddings_collection,
        behaviors_train_collection=behaviors_train_collection,
        behaviors_test_collection=behaviors_test_collection
    )

    news_embeddings_df = preprocess_news_embeddings(news_embeddings_df)
    user_profiles_df = create_user_profiles(behaviors_train_df, news_embeddings_df)

    # Build or load FAISS index
    try:
        index = load_faiss_index(INDEX_PATH)
        logger.info("FAISS index loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load FAISS index: {e}. Building a new index...")
        index = build_faiss_index(news_embeddings_collection, index_path=INDEX_PATH)

    # Compute recommendations using FAISS
    recommendations_df = compute_recommendations_faiss_incremental(
    user_profiles_df=user_profiles_df,
    news_embeddings_df=news_embeddings_df,  # Ensure this is a DataFrame
    index=index,
    top_k=TOP_K
    )

    # Save recommendations and evaluate as before
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
    ground_truth = load_ground_truth_parsed(MONGO_URI, DATABASE_NAME, behaviors_test_collection)
    
    # Evaluate performance
    metrics = evaluate_recommendations(recommendations_df, ground_truth, k=TOP_K)
    
    logger.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
        
    # Debug and insights
    #debug_and_insights(news_embeddings_df, behaviors_train_df, ground_truth, recommendations_df)

if __name__ == "__main__":
    main()

